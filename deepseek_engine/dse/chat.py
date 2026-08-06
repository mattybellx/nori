"""Nori — local chat UI with buttons; a Grok-style front-end for the engine.

Zero dependencies: stdlib ``http.server`` + a single-page app loaded from
``chat_page.html`` (editable without touching Python). Open a browser tab, type
a question, and see each strategy's answer as a card with an AI audit score,
Good/Bad/star buttons, a "better than baseline?" comparison, one-click "AI pick
best" and a side-by-side ghost compare. Everything is saved to ``sessions/``
exactly like the terminal ``ask`` tool.

Usage:
    python -m dse.chat                    # starts server + opens browser
    python -m dse.chat --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import ui
from . import ask as ask_mod
from .agent import Budget
from .config import EngineConfig, default_flags
from .env import load_env
from .freeform import FreeFormJudge, PromptTask
from .providers import PROVIDER_ENDPOINTS, make_provider
from .strategies import BestOfNAgent, ReactAgent, ReflexionAgent, SelfRefineAgent
from .telemetry import estimate_cost

_PAGE_FILE = Path(__file__).with_name("chat_page.html")
_PAGE_CACHE = {"mtime": 0.0, "html": ""}


def _load_page() -> str:
    """Load the single-page chat UI from disk (keeps the front-end editable)."""
    try:
        return _PAGE_FILE.read_text(encoding="utf-8")
    except OSError:
        return "<h1>chat_page.html is missing next to dse/chat.py</h1>"


_PAGE = _load_page()


def _page_html() -> str:
    """Serve the page, reloading from disk when it changes (mtime cache) so
    front-end edits don't require a server restart."""
    try:
        mtime = _PAGE_FILE.stat().st_mtime
    except OSError:
        return "<h1>chat_page.html is missing next to dse/chat.py</h1>"
    if _PAGE_CACHE["mtime"] != mtime:
        _PAGE_CACHE["mtime"] = mtime
        _PAGE_CACHE["html"] = _PAGE_FILE.read_text(encoding="utf-8")
    return _PAGE_CACHE["html"]


# ---------------------------------------------------------------------------
# User settings (bring-your-own-key) — persisted to settings.json
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "provider": "deepseek",
    "api_key": "",
    "base_url": "",
    "model_cheap": "deepseek-v4-flash",
    "model_expensive": "deepseek-v4-flash",
}

#: Settings file path; tests may monkeypatch this to a temp path.
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[:2] + "…"
    return key[:6] + "…" + key[-4:]


def load_settings() -> dict:
    """Read persisted user settings, merged over the defaults."""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {k: data.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS}


def save_settings(settings: dict) -> dict:
    merged = {k: settings.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS}
    try:
        SETTINGS_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    except OSError:
        pass  # settings persistence is best-effort; the session still works
    return merged


def detect_provider(base_url: str, provider: str = "") -> str:
    """Auto-detect a friendly provider display name from the endpoint URL."""
    if not base_url:
        return {"deepseek": "DeepSeek", "openai": "OpenAI", "ollama": "Ollama",
                "github": "GitHub Models"}.get(provider, "DeepSeek")
    host = base_url.lower()
    if "deepseek" in host:
        return "DeepSeek"
    if "openai" in host:
        return "OpenAI"
    if "github" in host:
        return "GitHub Models"
    if "azure" in host:
        return "Azure OpenAI"
    if "localhost" in host or "127.0.0.1" in host or "11434" in host:
        return "Ollama"
    return "Custom (OpenAI-compatible)"


# Thread-local so parallel strategy workers can label streamed LLM text.
_THREAD_LOCAL = threading.local()


class _StreamingLLM:
    """Wraps a real LLM and forwards every completion's FULL text to a callback
    as it completes, so the UI can stream the model's actual output live. The
    strategy label comes from a thread-local set by the worker thread."""

    def __init__(self, inner, on_text) -> None:
        self._inner = inner
        self._on_text = on_text

    def complete(self, messages, *, model="cheap", temperature=0.0, max_tokens=512):
        completion = self._inner.complete(
            messages, model=model, temperature=temperature, max_tokens=max_tokens)
        if completion.text and completion.text.strip():
            try:
                self._on_text(completion)
            except Exception:
                pass  # streaming is best-effort; never break the answer
        return completion


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "ChatServer"  # type: ignore[assignment]

    def log_message(self, format, *args):  # silence request logging
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            body = _page_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/ask":
            qs = parse_qs(parsed.query)
            self._stream_ask(
                qs.get("question", [""])[0],
                qs.get("mode", ["dev"])[0],
            )
            return
        if parsed.path == "/sessions":
            self._send_json(200, {"questions": ask_mod.recent_questions(limit=50)})
            return
        if parsed.path == "/stats":
            self._send_json(200, ask_mod.stats_json(ask_mod.load_records()))
            return
        if parsed.path == "/question":
            text = parse_qs(parsed.query).get("text", [""])[0]
            records = [
                r for r in ask_mod.load_records()
                if r.get("question") == text
            ]
            # keep only the most recent record per strategy so re-opened
            # history shows one clean card per strategy (records are ordered)
            latest: dict[str, dict] = {}
            for r in records:
                latest[str(r.get("strategy") or "")] = r
            self._send_json(200, {"question": text, "records": list(latest.values())})
            return
        if parsed.path == "/settings":
            s = load_settings()
            self._send_json(200, {
                "provider": s.get("provider", "deepseek"),
                "base_url": s.get("base_url", ""),
                "key_set": bool(s.get("api_key")),
                "key_masked": _mask_key(s.get("api_key", "")),
                "model_cheap": s.get("model_cheap"),
                "model_expensive": s.get("model_expensive"),
                "detected": detect_provider(s.get("base_url", ""), s.get("provider", "")),
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/rate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            updated = ask_mod.update_record(
                str(body.get("question") or ""), str(body.get("strategy") or ""),
                rating=body.get("rating"), comparison=body.get("comparison"),
            )
            self._send_json(200, {"updated": updated})
            return
        if parsed.path == "/settings":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            self._save_settings(body)
            return
        if parsed.path == "/synthesize":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            self._synthesize(body)
            return
        if parsed.path == "/pickbest":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            self._pick_best(body.get("question", ""), body.get("strategies"))
            return
        self._send_json(404, {"error": "not found"})

    # -- AI arbiter: pick the single best answer for a question ------------
    def _pick_best(self, question: str, strategies: list | None = None):
        """Ask the model to rank all saved answers and pick the best one.

        Returns the winning strategy plus a plain-language reason, so the UI
        can crown one card "BEST" and explain *why* in a side-by-side compare.
        ``strategies`` (optional) scopes the arbiter to only the strategies
        the current run actually produced, so it can never crown a card that
        isn't on screen.
        """
        records = [r for r in ask_mod.load_records() if r.get("question") == question]
        if strategies:
            wanted = {str(w) for w in strategies}
            records = [r for r in records if str(r.get("strategy")) in wanted]
        # keep only the most recent record per strategy (parallel runs + old
        # sessions can leave duplicates)
        latest: dict[str, dict] = {}
        for r in records:
            latest[str(r.get("strategy") or "")] = r
        records = list(latest.values())
        if not records:
            self._send_json(404, {"error": "no records for this question"})
            return
        parts = []
        for i, r in enumerate(records, 1):
            # flatten newlines (blank-completion trigger on V4 Flash)
            answer = " ".join((r.get("answer") or "").split())
            parts.append(f"[{i}] strategy={r.get('strategy')}\nANSWER:\n{answer}")
        prompt = (
            "Reply with EXACTLY three lines and nothing else (no preamble, "
            "no analysis, no markdown):\n"
            "BEST: <number of the best answer>\n"
            "REASON: <1-2 sentences on why it wins>\n"
            "WEAKNESS: <1 short sentence on what the best answer could still improve>\n\n"
            "QUESTION: {q}\n\n{answers}\n\n"
            "Pick the single best answer. You are a rigorous answer-quality "
            "arbiter. Judge accuracy, clarity, depth, and how directly each "
            "answer addresses the question."
        ).format(q=question, answers="\n\n".join(parts))
        models = self.server.pipeline["models"]
        llm = self.server.pipeline["llm"]
        text = ""
        try:
            for _ in range(3):  # V4 Flash occasionally returns blank -> retry
                text = llm.complete(
                    [{"role": "system", "content": prompt}],
                    model="expensive", max_tokens=600,
                ).text
                if text and text.strip():
                    break
        except Exception as exc:  # network / provider error
            self._send_json(502, {"error": f"pick-best failed: {exc}"})
            return
        import re as _re
        num = _re.search(r"BEST\s*:\s*(\d+)", text, _re.IGNORECASE)
        if not num:
            # the reasoning model rambled past the format: one tiny strict retry
            try:
                names = "\n".join(
                    f"[{i}] {r.get('strategy')}" for i, r in enumerate(records, 1))
                text2 = llm.complete(
                    [{"role": "system", "content": (
                        "Output ONLY the line: BEST: <number> (no other text).\n"
                        "Pick the single best answer.\n" + names)}],
                    model="expensive", max_tokens=50,
                ).text or ""
                num = _re.search(r"BEST\s*:\s*(\d+)", text2, _re.IGNORECASE)
            except Exception:
                num = None
        reason = _re.search(r"REASON\s*:\s*([^\n]+)", text, _re.IGNORECASE)
        weakness = _re.search(r"WEAKNESS\s*:\s*([^\n]+)", text, _re.IGNORECASE)

        def _clean_pick(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return ""
            if "<" in s or ">" in s:  # the model echoed the template placeholders
                return ""
            return s

        winner = None
        if num:
            idx = int(num.group(1)) - 1
            if 0 <= idx < len(records):
                winner = records[idx]
        if winner is None:
            # honest deterministic fallback: highest judge score wins
            scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]
            if scored:
                winner = max(scored, key=lambda r: r.get("judge_score") or 0)
        reason_txt = _clean_pick(reason.group(1).strip()[:600] if reason else "")
        weakness_txt = _clean_pick(weakness.group(1).strip()[:300] if weakness else "")
        if winner is not None and not reason_txt:
            reason_txt = ("Selected by highest judge score." if num is None
                          else "The AI judged it the best answer.")
        self._send_json(200, {
            "question": question,
            "winner": (winner or {}).get("strategy"),
            "reason": reason_txt,
            "weakness": weakness_txt,
            "raw": text[:400],
        })

    # -- user settings: bring-your-own-key with a live connection test ------
    def _save_settings(self, body: dict) -> None:
        """Validate a user-provided key/provider with a tiny completion; on
        success persist to settings.json and REBUILD the live pipeline so the
        very next question uses the new key (no restart)."""
        s = load_settings()
        if body.get("provider") in PROVIDER_ENDPOINTS:
            s["provider"] = str(body["provider"]).lower()
        if "api_key" in body:
            s["api_key"] = str(body.get("api_key") or "").strip()
        if "base_url" in body:
            s["base_url"] = str(body.get("base_url") or "").strip()
        if body.get("model_cheap"):
            s["model_cheap"] = str(body["model_cheap"]).strip()
        if body.get("model_expensive"):
            s["model_expensive"] = str(body["model_expensive"]).strip()
        detected = detect_provider(s["base_url"], s["provider"])
        error = None
        try:
            probe = build_pipeline(s["provider"], s["model_cheap"], s["model_expensive"],
                                   settings=s)
            probe["llm"].complete(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                model="cheap", max_tokens=8,
            )
        except Exception as exc:  # bad key / wrong URL / network
            error = str(exc)[:400]
        if error is None:
            save_settings(s)
            with self.server.lock:
                self.server.pipeline = build_pipeline(
                    s["provider"], s["model_cheap"], s["model_expensive"], settings=s)
            self._send_json(200, {
                "ok": True, "detected": detected, "provider": s["provider"],
                "key_masked": _mask_key(s["api_key"]),
                "message": f"Connection successful · {detected} — key saved",
            })
        else:
            # never persist a failing key (would brick the next server start)
            self._send_json(200, {
                "ok": False, "detected": detected, "provider": s["provider"],
                "error": error, "message": f"Connection failed · {detected}",
            })

    # -- answer synthesizer: merge the best parts of all candidates --------
    def _synthesize(self, body: dict) -> None:
        """Combine the strongest parts of every saved answer into ONE final
        polished version, with a short 'which parts came from where' note."""
        question = str(body.get("question") or "")
        records = [r for r in ask_mod.load_records() if r.get("question") == question]
        if not records:
            self._send_json(404, {"error": "no records for this question"})
            return
        requested = str(body.get("winner") or "")
        winner = next((r for r in records if r.get("strategy") == requested), None)
        if winner is None:
            scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]
            if scored:
                winner = max(scored, key=lambda r: r.get("judge_score") or 0)
        win_name = (winner or {}).get("strategy")
        parts = []
        for i, r in enumerate(records, 1):
            mark = "WINNER" if r.get("strategy") == win_name else "candidate"
            parts.append(f"[{i}] strategy={r.get('strategy')} ({mark})\n"
                         f"{' '.join((r.get('answer') or '').split())}")
        prompt = (
            "You are a careful answer synthesizer. Below are several candidate "
            "answers to the same question. Combine the STRONGEST parts of all "
            "candidates into ONE final, polished, complete answer. Keep the best "
            "explanations, examples, and wording; drop anything weaker or "
            "redundant.\n\n"
            "QUESTION: {q}\n\n{answers}\n\n"
            "Reply with exactly two sections and nothing else:\n"
            "FINAL ANSWER:\n<the merged answer>\n"
            "PARTS:\n<one short paragraph: which candidate(s) contributed which parts>"
        ).format(q=question, answers="\n\n".join(parts))
        text = ""
        try:
            llm = self.server.pipeline["llm"]
            for _ in range(3):
                text = llm.complete(
                    [{"role": "system", "content": prompt}],
                    model="expensive", max_tokens=1200,
                ).text
                if text and text.strip():
                    break
        except Exception as exc:  # network / provider error
            self._send_json(502, {"error": f"synthesize failed: {exc}"})
            return
        import re as _re
        fa = _re.search(r"FINAL ANSWER\s*:\s*(.*?)(?:\nPARTS\s*:|$)", text,
                        _re.DOTALL | _re.IGNORECASE)
        note = _re.search(r"PARTS\s*:\s*(.*)$", text, _re.DOTALL | _re.IGNORECASE)
        answer = (fa.group(1).strip() if fa else text.strip()) or ""
        if not answer:
            answer = "Synthesis returned an empty answer — please retry."
        if body.get("save_winner") and win_name:
            ask_mod.update_record(question, win_name, synthesized=answer)
        self._send_json(200, {
            "ok": True,
            "answer": answer[:8000],
            "note": (note.group(1).strip()[:1200] if note else ""),
            "winner": win_name,
        })

    # -- the SSE ask stream -------------------------------------------------
    def _stream_ask(self, question: str, mode: str = "dev"):
        if not question:
            self._send_json(400, {"error": "question is required"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        emit_lock = threading.Lock()

        def emit(event: str, data) -> None:
            with emit_lock:
                try:
                    self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        pipeline = self.server.pipeline
        all_strategies = pipeline["strategies"]
        models, budget = pipeline["models"], pipeline["budget"]
        # Auto mode runs a LEAN set (baseline + best repair) so it answers
        # fast; Dev mode runs all four. Evidence: reflexion beats single-shot
        # with p=0.000062; best_of_n adds zero.
        if mode == "auto":
            strategies = [s for s in all_strategies if s.name in ("react", "reflexion")]
        else:
            strategies = list(all_strategies)
        if not strategies:
            strategies = all_strategies
        n = len(strategies)
        emit("progress", {"done": 0, "total": n, "message": "starting…"})
        with self.server.lock:
            task = PromptTask(id=f"q{self.server.qid}", prompt=question)
            self.server.qid += 1

        # wrap the shared LLM so every completion's full text streams to the UI
        inner_llm = pipeline["llm"]

        def on_text(completion) -> None:
            emit("thinking", {
                "strategy": getattr(_THREAD_LOCAL, "strategy", "?"),
                "model": completion.model,
                "reasoning": getattr(completion, "reasoning", "") or "",
                "text": completion.text,
            })

        stream_llm = _StreamingLLM(inner_llm, on_text)
        for s in strategies:
            s.llm = stream_llm

        records: list[dict] = []
        results_lock = threading.Lock()

        def worker(index: int, strategy) -> None:
            _THREAD_LOCAL.strategy = strategy.name
            emit("progress", {"done": index, "total": n,
                              "message": f"{strategy.name}: working…"})
            try:
                result = strategy.solve(task, budget)
            except Exception as exc:
                emit("progress", {"done": index + 1, "total": n,
                                  "message": f"{strategy.name}: failed — {str(exc)[:80]}"})
                return
            meta = getattr(result, "verifier_meta", {}) or {}
            jscore = meta.get("judge_score")
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "strategy": strategy.name,
                "answer": result.answer,
                "judge_score": round(jscore, 1) if jscore is not None else None,
                "passed": result.success,
                "latency_s": round(result.latency_s, 2),
                "tokens": result.tokens_total,
                "cost_usd": round(estimate_cost(result, models), 6),
                "rating": None,
                "comparison": None,
            }
            with results_lock:
                records.append(record)
            emit("answer", record)
            emit("progress", {"done": index + 1, "total": n,
                              "message": f"{strategy.name}: done"})

        threads = [threading.Thread(target=worker, args=(i, s), daemon=True)
                   for i, s in enumerate(strategies)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for s in strategies:
            s.llm = inner_llm  # restore so /pickbest, /synthesize stay unwrapped
        # deterministic session order regardless of which strategy finished first
        _order = {"react": 0, "best_of_n": 1, "reflexion": 2, "self_refine": 3}
        records.sort(key=lambda r: _order.get(str(r.get("strategy") or ""), 99))
        with self.server.lock:
            path = ask_mod.save_records(records)
        emit("done", {"records": len(records), "file": path.name})


class ChatServer(ThreadingHTTPServer):
    def __init__(self, addr, handler, pipeline, lock):
        super().__init__(addr, handler)
        self.pipeline = pipeline
        self.lock = lock
        self.qid = 0


def build_pipeline(provider: str, model: str | None, judge_model: str | None,
                   settings: dict | None = None):
    load_env()
    s = settings if settings is not None else load_settings()
    if s.get("api_key"):
        # a user-configured key from the settings panel wins over CLI/env
        provider = s.get("provider") or provider or "deepseek"
        model = (s.get("model_cheap") or model
                 or os.environ.get("DSE_MODEL_CHEAP", "deepseek-v4-flash"))
        judge = s.get("model_expensive") or judge_model or model
    else:
        model = model or os.environ.get("DSE_MODEL_CHEAP", "deepseek-v4-flash")
        judge = judge_model or model
    from .config import provider_models

    models = provider_models(cheap_model=model, expensive_model=judge)
    llm = make_provider(
        provider, models=models,
        api_key=(s.get("api_key") or None),
        base_url=(s.get("base_url") or None),
    )
    if llm is None:
        raise ValueError("chat requires a real provider (deepseek/ollama/...); got 'mock'")
    config = EngineConfig(seed=0, flags=default_flags())
    judge_fn = FreeFormJudge(llm, model="expensive")
    strategies = [
        ReactAgent(llm, judge_fn, config, models),
        BestOfNAgent(llm, judge_fn, config, models, n=3),
        ReflexionAgent(llm, judge_fn, config, models),
        SelfRefineAgent(llm, judge_fn, config, models),
    ]
    budget = Budget(max_trials=config.max_trials, max_search_nodes=config.max_search_nodes)
    return {"models": models, "strategies": strategies, "budget": budget, "llm": llm}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nori - local chat UI")
    parser.add_argument("--provider", default="deepseek",
                        choices=["deepseek", "ollama", "openai", "github"])
    parser.add_argument("--model", help="answer model (cheap tier)")
    parser.add_argument("--judge-model", help="judge model (expensive tier)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    pipeline = build_pipeline(args.provider, args.model, args.judge_model,
                              settings=load_settings())
    models = pipeline["models"]

    # find a free port near the requested one
    port = args.port
    server = None
    for candidate in range(port, port + 20):
        try:
            server = ChatServer(("127.0.0.1", candidate), ChatHandler, pipeline, threading.Lock())
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        ui.fail(f"could not bind any port in {port}-{port+20}")
        return 1

    url = f"http://127.0.0.1:{port}"
    print(ui.ok(f"Nori running at {url}"))
    print(ui.dim(f"answer model={models['cheap'].provider_model}  "
                 f"judge model={models['expensive'].provider_model}"))
    print(ui.dim("press Ctrl+C to stop"))
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(ui.dim("\nstopped"))
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
