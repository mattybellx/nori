"""Composable architecture primitives (Phase 1 of the discovery spec, §5).

A primitive is a callable ``(ctx, node, inputs) -> NodeOutput``. Primitives
wrap the project's EXISTING infrastructure (llm.complete, verifier.score, the
never-worse guards, the calibrated judge, the strategies themselves) so new
architectures are built from tested pieces — never from new untrusted code.

Node outputs carry provenance: kind, tokens, latency, model, verdict. This is
the raw material for the fitness record (§17) and honest cost accounting (§13).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agent import Budget
from ..environment import Task
from ..guards import robust_score, selection_guard, synthesis_guard
from ..verifier import Verdict

# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


@dataclass
class ExecutionContext:
    """Everything a primitive may need. Built once per graph execution.

    ``agents`` is a ``{name: Agent}`` map so the ``strategy`` primitive can
    run whole existing strategies as single nodes (the baseline set, §6).
    """

    llm: Any = None
    verifier: Any = None
    models: dict | None = None
    config: Any = None
    agents: dict[str, Any] = field(default_factory=dict)
    task: Task | None = None
    budget: Budget | None = None
    cost_per_1m_in: float = 0.0
    cost_per_1m_out: float = 0.0
    # set by the executor during a run
    outputs: dict[str, "NodeOutput"] = field(default_factory=dict)
    t_start: float = field(default_factory=time.perf_counter)

    def system_prompt(self, extra: str = "") -> str:
        """Match BaseAgent._system so MockLLM resolves the task by TASK_ID."""
        task = self.task
        tid = getattr(task, "id", "?")
        return f"TASK_ID: {tid}\n{extra}".rstrip()

    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1e6 * self.cost_per_1m_in + tokens_out / 1e6 * self.cost_per_1m_out


@dataclass
class NodeOutput:
    """Output of one primitive invocation, with provenance."""

    value: Any
    kind: str = "data"
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.value if isinstance(self.value, str) else ""


Primitive = Callable[[ExecutionContext, Any, dict[str, NodeOutput]], NodeOutput]

_REGISTRY: dict[str, Primitive] = {}


def register(name: str) -> Callable[[Primitive], Primitive]:
    def deco(fn: Primitive) -> Primitive:
        if name in _REGISTRY:
            raise ValueError(f"duplicate primitive {name!r}")
        _REGISTRY[name] = fn
        return fn
    return deco


def primitive(name: str) -> Primitive:
    if name not in _REGISTRY:
        raise KeyError(f"unknown primitive {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


def gather_values(inputs: dict[str, NodeOutput]) -> list[Any]:
    """Flatten a node's inputs into an ordered list (edge declaration order)."""
    return [o.value for o in inputs.values()]


def _llm_call(ctx: ExecutionContext, node, system: str, user: str,
              model: str | None, max_tokens: int) -> tuple[str, int, int, float, str]:
    tier = model or node.params.get("model", "cheap")
    cfg = (ctx.models or {}).get(tier)
    if cfg is None:
        raise ValueError(f"model tier {tier!r} not configured")
    comp = ctx.llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=tier,
        max_tokens=max_tokens,
    )
    return (comp.text or "", comp.tokens_in, comp.tokens_out, comp.latency_s, comp.model)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@register("generate")
def _gen(ctx, node, inputs):
    task = ctx.task
    user = getattr(task, "prompt", "")
    extra = node.params.get("extra", "")
    text, ti, to, lat, model = _llm_call(
        ctx, node, ctx.system_prompt(extra), user, node.params.get("model"),
        int(node.params.get("max_tokens", getattr(ctx.config, "max_tokens_per_call", 2048))))
    return NodeOutput(text, kind="generate", tokens_in=ti, tokens_out=to,
                      latency_s=lat, model=model)


@register("sample_n")
def _sample_n(ctx, node, inputs):
    n = int(node.params.get("n", 4))
    model = node.params.get("model")
    texts, ti, to, lat = [], 0, 0, 0.0
    for _ in range(n):
        t, i, o, l, m = _llm_call(ctx, node, ctx.system_prompt("Produce a draft."),
                                  getattr(ctx.task, "prompt", ""), model, 2048)
        texts.append(t); ti += i; to += o; lat += l
    return NodeOutput(texts, kind="sample_n", tokens_in=ti, tokens_out=to,
                      latency_s=lat, model=model or "cheap")


# ---------------------------------------------------------------------------
# Verification & judging
# ---------------------------------------------------------------------------


@register("verify")
def _verify(ctx, node, inputs):
    answer = inputs.get("answer", inputs.get(node.id))
    if answer is None and inputs:
        answer = next(iter(inputs.values())).value
    text = answer if isinstance(answer, str) else (answer.text if hasattr(answer, "text") else str(answer))
    verdict = ctx.verifier.score(text, ctx.task)
    return NodeOutput(verdict, kind="verify",
                      meta={"score": verdict.score, "passed": verdict.passed,
                            "details": dict(verdict.details or {})})


def _judge_callable(ctx, node):
    """A ``(text) -> float|None`` judge for robust_score — the project's
    calibrated FreeFormJudge (median-N de-noising via guards.robust_score)."""
    from ..freeform import FreeFormJudge

    judge = FreeFormJudge(ctx.llm, model=node.params.get("judge_model", "expensive"))
    task = ctx.task

    def fn(text: str) -> float | None:
        try:
            v = judge.score(text or "", task)
            return v.details.get("judge_score")
        except Exception:
            return None
    return fn


@register("judge")
def _judge(ctx, node, inputs):
    target = next(iter(inputs.values())).value
    samples = int(node.params.get("samples", 3))
    fn = _judge_callable(ctx, node)
    if isinstance(target, list):
        score = robust_score(fn, "\n".join(str(t) for t in target), samples=samples)
    else:
        score = robust_score(fn, str(target), samples=samples)
    return NodeOutput(score, kind="judge", meta={"score": score, "samples": samples})


@register("verify_items")
def _verify_items(ctx, node, inputs):
    """Score every incoming item with the OBJECTIVE verifier (like the real
    best_of_n agent: N drafts, verifier-selected). Each input may be a
    strategy RunResult, a dict or plain text. Returns
    ``[{"id","text","score","passed"}]`` where score is the verifier score."""
    items = []
    for port, out in inputs.items():
        v = out.value
        if hasattr(v, "answer"):
            text = v.answer
        elif isinstance(v, dict):
            text = v.get("answer") or v.get("text") or str(v)
        else:
            text = str(v)
        verdict = ctx.verifier.score(text, ctx.task)
        items.append({"id": port, "text": text, "score": verdict.score,
                      "passed": bool(verdict.passed)})
    return NodeOutput(items, kind="verify_items", meta={"n": len(items)})


@register("score_items")
def _score_items(ctx, node, inputs):
    """Score every incoming item (each input = one candidate) -> list of
    ``{"id","text","score"}`` dicts. Mirrors the production pipeline's per-
    candidate judging. Handles strategy RunResults, dicts and plain text."""
    fn = _judge_callable(ctx, node)
    samples = int(node.params.get("samples", 3))
    items = []
    for port, out in inputs.items():
        v = out.value
        if hasattr(v, "answer"):
            text = v.answer
        elif isinstance(v, dict):
            text = v.get("answer") or v.get("text") or str(v)
        else:
            text = str(v)
        score = robust_score(fn, text, samples=samples)
        items.append({"id": port, "text": text, "score": score})
    return NodeOutput(items, kind="score_items", meta={"n": len(items)})


@register("select_best")
def _select_best(ctx, node, inputs):
    items = next(iter(inputs.values())).value
    scored = [it for it in items if it.get("score") is not None]
    if not scored:
        return NodeOutput(items[0] if items else {}, kind="select_best", meta={"chosen": None})
    best = max(scored, key=lambda it: it["score"])
    return NodeOutput(best, kind="select_best", meta={"chosen": best["id"],
                                                      "score": best["score"]})


@register("selection_guard")
def _select_guard(ctx, node, inputs):
    """Reuse guards.selection_guard: never ship a candidate the judge scored
    below the baseline by more than the noise floor (§19 never-worse)."""
    items = next(iter(inputs.values())).value
    records = [{"strategy": it["id"], "judge_score": it.get("score")} for it in items]
    baseline = node.params.get("baseline", "react")
    floor = float(node.params.get("floor", 0.5))
    chosen = selection_guard(records, baseline, noise_floor=floor)
    chosen_item = next(it for it in items if it["id"] == chosen)
    return NodeOutput(chosen_item, kind="selection_guard",
                      meta={"chosen": chosen, "fired": chosen != baseline})


# ---------------------------------------------------------------------------
# Synthesis (reuses the production best-of-all merge + no-regression guard)
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = (
    "You are a careful answer synthesizer. Combine the STRONGEST parts of the "
    "candidate answers into ONE final, polished, complete answer. Keep the best "
    "explanations, examples, and wording; drop anything weaker or redundant.\n"
    "QUESTION: {question}\n\n{answers}\n\n"
    "Reply with exactly two sections and nothing else:\n"
    "FINAL ANSWER:\n<the merged answer>\n"
    "PARTS:\n<one short paragraph: which candidate(s) contributed which parts>"
)


@register("synthesize")
def _synthesize(ctx, node, inputs):
    candidates = gather_values(inputs)
    candidates = [c["text"] if isinstance(c, dict) and "text" in c else str(c)
                  for c in candidates]
    if not candidates:
        return NodeOutput("", kind="synthesize")
    parts = "\n\n".join(f"CANDIDATE {i+1}:\n{c}" for i, c in enumerate(candidates))
    prompt = _SYNTH_SYSTEM.format(question=getattr(ctx.task, "prompt", ""), answers=parts)
    text = ""
    ti = to = 0
    lat = 0.0
    for _ in range(3):
        try:
            text, i, o, l, m = _llm_call(ctx, node, prompt, "", "expensive", 1200)
            ti, to, lat = i, o, l
            if text.strip():
                break
        except Exception:
            continue
    match = re.search(r"FINAL ANSWER\s*:\s*(.*?)(?:\nPARTS\s*:|$)", text, re.DOTALL | re.IGNORECASE)
    merged = match.group(1).strip() if match else text.strip()
    return NodeOutput(merged, kind="synthesize", tokens_in=ti, tokens_out=to,
                      latency_s=lat, model="expensive")


@register("synthesis_guard")
def _synth_guard(ctx, node, inputs):
    """Reuse guards.synthesis_guard — the never-worse no-regression gate:
    ship the merge only if it is grounded in a candidate AND scored at least
    as well as the winner; otherwise fall back to the winner verbatim."""
    synth = inputs.get("synth").value if "synth" in inputs else None
    winner = inputs.get("winner").value if "winner" in inputs else None
    cands = inputs.get("candidates")
    candidates = list(cands.value) if cands is not None else []
    if synth is None or winner is None:
        raise ValueError("synthesis_guard needs 'synth' and 'winner' inputs")
    candidates = [c["text"] if isinstance(c, dict) and "text" in c else str(c)
                  for c in candidates]
    winner_text = winner["text"] if isinstance(winner, dict) and "text" in winner else str(winner)
    judge = _judge_callable(ctx, node)
    final, used, why = synthesis_guard(
        synth if isinstance(synth, str) else str(synth),
        candidates, winner_text, judge=judge,
        min_overlap=float(node.params.get("min_overlap", 0.10)),
        score_margin=float(node.params.get("score_margin", 0.5)))
    return NodeOutput(final, kind="synthesis_guard",
                      meta={"used_synth": used, "why": why, "winner": winner_text})


# ---------------------------------------------------------------------------
# Aggregation & control
# ---------------------------------------------------------------------------


@register("gather")
def _gather(ctx, node, inputs):
    return NodeOutput(gather_values(inputs), kind="gather", meta={"n": len(inputs)})


@register("identity")
def _identity(ctx, node, inputs):
    out = next(iter(inputs.values()))
    return NodeOutput(out.value, kind="identity", meta=dict(out.meta))


@register("route_disagreement")
def _route_disagreement(ctx, node, inputs):
    """Conditional routing (§7): measure disagreement across incoming
    candidates and set ``meta["__route"]`` to ``high`` or ``low``. The
    executor follows only the out-edge whose ``port`` matches."""
    values = gather_values(inputs)
    scores = [v.get("score") for v in values if isinstance(v, dict) and v.get("score") is not None]
    if len(scores) >= 2:
        spread = max(scores) - min(scores)
    else:
        spread = 0.0
    threshold = float(node.params.get("threshold", 1.0))
    route = "high" if spread > threshold else "low"
    return NodeOutput(values, kind="route_disagreement",
                      meta={"__route": route, "spread": spread, "threshold": threshold})


@register("strategy")
def _strategy(ctx, node, inputs):
    """Run a whole existing strategy as a single node (the baseline set, §6).
    Reuses the exact Agent.solve() implementation — maximum reuse."""
    name = node.params.get("name")
    if name not in ctx.agents:
        raise ValueError(f"strategy {name!r} not in context; have {sorted(ctx.agents)}")
    rr = ctx.agents[name].solve(ctx.task, ctx.budget)
    return NodeOutput(rr, kind="strategy", tokens_in=rr.tokens_in, tokens_out=rr.tokens_out,
                      latency_s=rr.latency_s, model=rr.route_tier,
                      meta={"strategy": name, "success": rr.success,
                            "score": rr.verifier_score, "answer": rr.answer,
                            "attempts": rr.attempts})


@register("extract")
def _extract(ctx, node, inputs):
    """Pull the answer text out of a strategy RunResult, a scored item dict
    (``{id, text, score}`` from score_items/select_best), or pass strings
    through."""
    out = next(iter(inputs.values())).value
    if hasattr(out, "answer"):
        text = out.answer
    elif isinstance(out, dict) and "answer" in out:
        text = out["answer"]
    elif isinstance(out, dict) and "text" in out:
        text = out["text"]
    else:
        text = str(out)
    return NodeOutput(text, kind="extract", meta={"strategy": getattr(out, "strategy", "")})
