"""Tests for the terminal UI helpers, free-form judge, ask tool, and progress."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dse import ui
from dse.freeform import FreeFormJudge, PromptTask, parse_grade
from dse.providers import provider_models


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def test_color_disabled_without_tty(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.red("x") == "x"
    # off a TTY, the UI falls back to ASCII glyphs (never mojibake)
    assert ui.ok("y") == "OK y"
    assert ui.fail("z") == "X z"
    assert ui.warn("w") == "! w"


def test_table_aligns_and_styles():
    headers = ["name", "score"]
    rows = [["alpha", "0.9"], ["longer-name", "0.1"]]
    out = ui.table(headers, rows, header_style=ui.bold)
    lines = out.splitlines()
    assert lines[0].startswith("name")
    # both columns aligned (plain text, style stripped for the styled header)
    assert "0.9" in lines[2]
    assert "0.1" in lines[3]


def test_progress_non_tty_finishes(monkeypatch, capsys):
    update, finish = ui.progress(10, "t", width=5)
    update(5, "step")
    finish("all done")
    captured = capsys.readouterr().out
    assert "10 done" in captured
    assert "all done" in captured


# ---------------------------------------------------------------------------
# Free-form judge
# ---------------------------------------------------------------------------
def test_parse_grade():
    score, feedback = parse_grade("SCORE: 7/10\nFEEDBACK: close but missing detail")
    assert score == 7.0
    assert "missing detail" in feedback
    score2, _ = parse_grade("blah")
    assert score2 is None
    # scores are clamped to 0..10
    s, _ = parse_grade("SCORE: 42")
    assert s == 10.0


def test_parse_grade_fraction_and_letter():
    # rubric-style "Score: 35/40" is scaled to 8.75
    assert parse_grade("Score: 35/40")[0] == pytest.approx(8.75)
    # letter grades (deepseek-reasoner style)
    assert parse_grade("Grade: A")[0] == pytest.approx(9.2)
    assert parse_grade("grade: B+")[0] == pytest.approx(8.2)
    assert parse_grade("Grade: F")[0] == pytest.approx(1.0)


class _ScriptedJudgeLLM:
    """Returns canned SCORE/FEEDBACK responses for FreeFormJudge tests."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, messages, *, model="expensive", temperature=0.0, max_tokens=200):
        from dse.llm import Completion

        text = self._replies[self.calls % len(self._replies)]
        self.calls += 1
        return Completion(text=text, tokens_in=20, tokens_out=10, latency_s=0.05, model=model)


def test_freeform_judge_parses_score_and_feedback():
    llm = _ScriptedJudgeLLM(["SCORE: 8/10\nFEEDBACK: solid reasoning"])
    judge = FreeFormJudge(llm, model="expensive")
    task = PromptTask(id="q1", prompt="why is the sky blue?")
    verdict = judge.score("Because of Rayleigh scattering.", task)
    assert verdict.passed and abs(verdict.score - 0.8) < 1e-9
    assert verdict.details["judge_score"] == 8.0
    assert "solid reasoning" in verdict.details["feedback"]


def test_freeform_judge_passes_question_and_answer_in_prompt():
    seen = {}

    class _Capture(_ScriptedJudgeLLM):
        def complete(self, messages, *, model="expensive", temperature=0.0, max_tokens=200):
            seen["system"] = messages[0]["content"]
            return super().complete(messages, model=model)

    judge = FreeFormJudge(_Capture(["SCORE: 5"]) , model="expensive")
    task = PromptTask(id="q9", prompt="What is 2+2?")
    judge.score("4", task)
    assert "q9" in seen["system"]
    assert "What is 2+2?" in seen["system"]
    assert "4" in seen["system"]


# ---------------------------------------------------------------------------
# Ask tool (one-shot against a scripted OpenAI-compatible server)
# ---------------------------------------------------------------------------
def _scripted_server():
    """Serves canned completions: answers + a judge grade."""
    captured = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            captured["n"] += 1
            system = body["messages"][0]["content"]
            if "MODE: grade" in system:
                content = "SCORE: 7/10\nFEEDBACK: good answer"
            elif "Pick the single best answer" in system:
                content = "BEST: 1\nREASON: the first answer is the most accurate and clear\nWEAKNESS: could add one example"
            else:
                content = "The sky looks blue because of Rayleigh scattering."
            payload = {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, captured


def test_ask_one_shot_writes_session(monkeypatch, tmp_path):
    from dse import ask

    server, captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        monkeypatch.setenv("DSE_MODEL_CHEAP", "deepseek-chat")
        monkeypatch.setenv("DSE_MODEL_EXPENSIVE", "deepseek-reasoner")
        monkeypatch.setattr(ask, "SESSIONS_DIR", tmp_path / "sessions")

        rc = ask.main(["--provider", "deepseek", "--question", "why is the sky blue?"])
        assert rc == 0
        assert captured["n"] > 0  # real API calls were made through the client

        records = ask.load_records()
        assert len(records) == 4  # react, best_of_n, reflexion, self_refine
        assert {r["strategy"] for r in records} == {"react", "best_of_n", "reflexion", "self_refine"}
        for record in records:
            assert record["question"] == "why is the sky blue?"
            assert "Rayleigh" in record["answer"]
            assert record["judge_score"] == 7.0
            assert record["rating"] is None
    finally:
        server.shutdown()


def test_ask_stats_empty(tmp_path, capsys):
    from dse import ask

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ask, "SESSIONS_DIR", tmp_path / "nope")
    ask.print_stats([])
    captured = capsys.readouterr().out
    assert "no sessions yet" in captured
    monkeypatch.undo()


def test_ask_requires_real_provider():
    from dse import ask

    with pytest.raises(ValueError, match="real provider"):
        ask.build_ask_pipeline("mock", None, None)


def test_update_record_persists_rating(tmp_path):
    from dse import ask

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ask, "SESSIONS_DIR", tmp_path / "sessions")
    try:
        ask.save_records([{"question": "hello", "strategy": "react", "answer": "x", "rating": None}])
        updated = ask.update_record("hello", "react", rating=5, comparison="better")
        assert updated == 1
        loaded = ask.load_records()
        assert loaded[0]["rating"] == 5
        assert loaded[0]["comparison"] == "better"
    finally:
        monkeypatch.undo()


def test_chat_server_endpoints(tmp_path, monkeypatch):
    """The chat UI: GET / serves the page, GET /ask streams SSE answers, and
    POST /rate persists button clicks to the session file."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod
    from dse import chat as chat_mod

    server, captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        pipeline = chat_mod.build_pipeline("deepseek", None, None)
        monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")

        srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            port = srv.server_address[1]
            # page
            html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
            assert "nori" in html
            # SSE ask stream
            url = f"http://127.0.0.1:{port}/ask?question=" + urllib.parse.quote("why is the sky blue?")
            body = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
            assert "event: answer" in body
            assert "react" in body and "self_refine" in body
            assert "event: done" in body
            # rate button
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/rate",
                data=json.dumps({"question": "why is the sky blue?", "strategy": "react", "rating": 5}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
            assert resp["updated"] >= 1
            # and the session file reflects it
            loaded = ask_mod.load_records()
            assert any(r["strategy"] == "react" and r["rating"] == 5 for r in loaded)
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_chat_history_endpoints(tmp_path, monkeypatch):
    """/sessions lists distinct recent questions and /question returns the
    full record set for one question (chat sidebar + history re-open)."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod
    from dse import chat as chat_mod

    server, captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        pipeline = chat_mod.build_pipeline("deepseek", None, None)
        monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")

        srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            port = srv.server_address[1]
            # ask two different questions so history has entries
            for q in ("what is 2+2?", "why is the sky blue?"):
                url = f"http://127.0.0.1:{port}/ask?question=" + urllib.parse.quote(q)
                urllib.request.urlopen(url, timeout=30).read()

            # /sessions -> distinct questions, most recent first
            sessions = json.loads(
                urllib.request.urlopen(f"http://127.0.0.1:{port}/sessions", timeout=10).read().decode("utf-8")
            )
            assert sessions["questions"][0] == "why is the sky blue?"  # most recent first
            assert set(sessions["questions"]) == {"what is 2+2?", "why is the sky blue?"}

            # /question?text= -> all records for one question
            url = f"http://127.0.0.1:{port}/question?text=" + urllib.parse.quote("what is 2+2?")
            detail = json.loads(urllib.request.urlopen(url, timeout=10).read().decode("utf-8"))
            assert detail["question"] == "what is 2+2?"
            assert {r["strategy"] for r in detail["records"]} == {"react", "best_of_n", "reflexion", "self_refine"}
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_pick_best_endpoint(tmp_path, monkeypatch):
    """POST /pickbest asks the model to rank the saved answers and returns the
    winning strategy plus a plain-language reason."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod
    from dse import chat as chat_mod

    server, captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        pipeline = chat_mod.build_pipeline("deepseek", None, None)
        monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")

        srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            port = srv.server_address[1]
            # create records for the question first
            url = f"http://127.0.0.1:{port}/ask?question=" + urllib.parse.quote("why is the sky blue?")
            urllib.request.urlopen(url, timeout=30).read()

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/pickbest",
                data=json.dumps({"question": "why is the sky blue?"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))
            assert resp["winner"] == "react"  # scripted server always says BEST: 1
            assert "accurate and clear" in resp["reason"]
            assert resp["weakness"]
            # unknown question -> 404
            req2 = urllib.request.Request(
                f"http://127.0.0.1:{port}/pickbest",
                data=json.dumps({"question": "nope"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req2, timeout=10)
                raise AssertionError("expected 404 for unknown question")
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_settings_endpoint(tmp_path, monkeypatch):
    """GET /settings returns masked key info; POST /settings tests the
    connection with a tiny completion, saves on success, and never leaks the
    full key back to the page."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod
    from dse import chat as chat_mod

    server, captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        pipeline = chat_mod.build_pipeline("deepseek", None, None)
        monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")
        monkeypatch.setattr(chat_mod, "SETTINGS_FILE", tmp_path / "settings.json")

        srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            port = srv.server_address[1]
            root = f"http://127.0.0.1:{port}"

            # 1) fresh GET -> no key, deepseek default
            d = json.loads(urllib.request.urlopen(root + "/settings", timeout=10).read())
            assert d["provider"] == "deepseek"
            assert d["key_set"] is False
            assert d["key_masked"] == ""
            assert d["detected"] == "DeepSeek"

            # 2) POST a bad key against the scripted server -> still succeeds
            #    (scripted server answers anything) and reports ok
            req = urllib.request.Request(
                root + "/settings",
                data=json.dumps({"provider": "deepseek", "api_key": "sk-test-1234567890",
                                 "base_url": ""}).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            d = json.loads(urllib.request.urlopen(req, timeout=15).read())
            assert d["ok"] is True
            assert d["detected"] == "DeepSeek"
            assert d["key_masked"].startswith("sk-tes") and d["key_masked"].endswith("7890")

            # 3) GET now reflects the saved key but never returns the full key
            d = json.loads(urllib.request.urlopen(root + "/settings", timeout=10).read())
            assert d["key_set"] is True
            assert d["key_masked"] == "sk-tes…7890"
            assert "sk-test-1234567890" not in json.dumps(d)

            # 4) the settings file was persisted
            assert (tmp_path / "settings.json").is_file()
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_synthesize_endpoint(tmp_path, monkeypatch):
    """POST /synthesize merges the saved answers into one final version and
    echoes the winner."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod
    from dse import chat as chat_mod

    server, captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        pipeline = chat_mod.build_pipeline("deepseek", None, None)
        monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")

        srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            port = srv.server_address[1]
            root = f"http://127.0.0.1:{port}"
            url = root + "/ask?question=" + urllib.parse.quote("why is the sky blue?")
            urllib.request.urlopen(url, timeout=30).read()

            req = urllib.request.Request(
                root + "/synthesize",
                data=json.dumps({"question": "why is the sky blue?", "winner": "react"}).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            d = json.loads(urllib.request.urlopen(req, timeout=15).read())
            assert d["ok"] is True
            assert d["answer"]
            assert d["winner"] == "react"
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Never-worse guards on the LIVE HTTP paths
# ---------------------------------------------------------------------------
def _guard_scripted_server(arbiter_pick=1,
                           synth_text="The sky looks blue because of Rayleigh scattering and atmospheric scattering.",
                           judge_synth="SCORE: 3/10",
                           judge_winner="SCORE: 8/10",
                           judge_generic="SCORE: 7/10"):
    """Scripted provider with knobs for the never-worse guard tests.

    - ``arbiter_pick``: 1-based index the arbiter chooses in /pickbest.
    - ``synth_text``: what the synthesizer returns (must share bigrams with the
      winner answer to count as "grounded").
    - ``judge_synth`` / ``judge_winner``: replies for ``_score_text`` on the
      synthesis vs the winner answer. The judge prompt embeds the answer text,
      so the server tells them apart via ``synth_text[:50]``.
    """
    captured = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            captured["n"] += 1
            system = body["messages"][0]["content"]
            if "MODE: grade" in system:
                content = judge_generic
            elif "Pick the single best answer" in system:
                content = f"BEST: {arbiter_pick}\nREASON: the chosen answer is best\nWEAKNESS: minor"
            elif "calibrated answer grader" in system:
                if synth_text and synth_text[:50] in system:
                    content = judge_synth
                else:
                    content = judge_winner
            elif "answer synthesizer" in system:
                content = synth_text
            else:
                content = "The sky looks blue because of Rayleigh scattering."
            payload = {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, captured


def _post(root, path, body):
    import urllib.request

    req = urllib.request.Request(
        root + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))


def _start_guard_chat(server, tmp_path, monkeypatch):
    """Boot a ChatServer whose provider is ``server``; returns (srv, root)."""
    from dse import ask as ask_mod
    from dse import chat as chat_mod

    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("DSE_PROVIDER_URL", base)
    pipeline = chat_mod.build_pipeline("deepseek", None, None)
    monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")
    srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_synthesize_guard_ships_and_saves_winner(tmp_path, monkeypatch):
    """A grounded synthesis scored >= the winner ships (guard='shipped') and
    save_winner persists it onto the winner's record."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod

    server, _ = _guard_scripted_server(judge_synth="SCORE: 8/10", judge_winner="SCORE: 7/10")
    try:
        srv, root = _start_guard_chat(server, tmp_path, monkeypatch)
        try:
            url = root + "/ask?question=" + urllib.parse.quote("why is the sky blue?")
            urllib.request.urlopen(url, timeout=30).read()
            d = _post(root, "/synthesize",
                      {"question": "why is the sky blue?", "winner": "react", "save_winner": True})
            assert d["ok"] is True
            assert d["guard"] == "shipped"
            assert d["answer"] and "Rayleigh" in d["answer"]
            records = ask_mod.load_records()
            assert any(r["strategy"] == "react" and r.get("synthesized") for r in records)
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_synthesize_guard_falls_back_when_ungrounded(tmp_path, monkeypatch):
    """A synthesis sharing no bigrams with any candidate is ungrounded -> the
    guard falls back to the winner verbatim and save_winner is NOT persisted."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod

    server, _ = _guard_scripted_server(
        synth_text="Mars has two small moons and a thin carbon dioxide atmosphere.")
    try:
        srv, root = _start_guard_chat(server, tmp_path, monkeypatch)
        try:
            url = root + "/ask?question=" + urllib.parse.quote("why is the sky blue?")
            urllib.request.urlopen(url, timeout=30).read()
            d = _post(root, "/synthesize",
                      {"question": "why is the sky blue?", "winner": "react", "save_winner": True})
            assert d["ok"] is True
            assert d["guard"] == "fell_back"
            # final answer IS the winner's original answer, verbatim
            assert d["answer"] == "The sky looks blue because of Rayleigh scattering."
            assert "never-worse guard" in d["note"]
            assert not any(r.get("synthesized") for r in ask_mod.load_records())
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_synthesize_guard_falls_back_when_scored_below_winner(tmp_path, monkeypatch):
    """A grounded synthesis the judge scores below the winner by more than the
    margin is rejected: guard='fell_back', answer is the winner's, not saved."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod

    server, _ = _guard_scripted_server(
        synth_text="The sky looks blue because of Rayleigh scattering and atmospheric scattering.",
        judge_synth="SCORE: 3/10", judge_winner="SCORE: 8/10")
    try:
        srv, root = _start_guard_chat(server, tmp_path, monkeypatch)
        try:
            url = root + "/ask?question=" + urllib.parse.quote("why is the sky blue?")
            urllib.request.urlopen(url, timeout=30).read()
            d = _post(root, "/synthesize",
                      {"question": "why is the sky blue?", "winner": "react", "save_winner": True})
            assert d["ok"] is True
            assert d["guard"] == "fell_back"
            assert d["answer"] == "The sky looks blue because of Rayleigh scattering."
            assert "synthesis 3.0 < winner 8.0" in d["note"]
            assert not any(r.get("synthesized") for r in ask_mod.load_records())
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_pickbest_selection_guard_keeps_baseline(tmp_path, monkeypatch):
    """When the arbiter picks a non-baseline the judge scored below react by
    more than the noise floor, the selection guard crowns react instead."""
    from dse import ask as ask_mod

    server, _ = _guard_scripted_server(arbiter_pick=2)
    try:
        srv, root = _start_guard_chat(server, tmp_path, monkeypatch)
        try:
            ask_mod.save_records([
                {"question": "why is the sky blue?", "strategy": "react",
                 "answer": "The sky looks blue because of Rayleigh scattering.", "judge_score": 8.0},
                {"question": "why is the sky blue?", "strategy": "self_refine",
                 "answer": "Because of scattering of light by air molecules.", "judge_score": 4.0},
            ])
            d = _post(root, "/pickbest",
                      {"question": "why is the sky blue?", "strategies": ["react", "self_refine"]})
            assert d["winner"] == "react"
            assert "never-worse guard" in d["reason"]
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_pickbest_selection_guard_allows_strong_winner(tmp_path, monkeypatch):
    """When the arbiter's pick clears the noise floor vs react, the selection
    guard lets it through (no guard note)."""
    from dse import ask as ask_mod

    server, _ = _guard_scripted_server(arbiter_pick=2)
    try:
        srv, root = _start_guard_chat(server, tmp_path, monkeypatch)
        try:
            ask_mod.save_records([
                {"question": "why is the sky blue?", "strategy": "react",
                 "answer": "The sky looks blue because of Rayleigh scattering.", "judge_score": 8.0},
                {"question": "why is the sky blue?", "strategy": "self_refine",
                 "answer": "Because of scattering of light by air molecules.", "judge_score": 7.5},
            ])
            d = _post(root, "/pickbest",
                      {"question": "why is the sky blue?", "strategies": ["react", "self_refine"]})
            assert d["winner"] == "self_refine"
            assert "never-worse guard" not in d["reason"]
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_auth_endpoints_unconfigured(tmp_path, monkeypatch):
    """/auth/status reports not-signed-in, /auth/google 400s without
    credentials, and /auth/signout is a safe no-op."""
    import urllib.error
    import urllib.request

    from dse import ask as ask_mod
    from dse import chat as chat_mod

    server, _captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        # isolate settings + token store so Google is definitely unconfigured
        monkeypatch.setattr(chat_mod, "SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setattr(chat_mod.google_auth, "AUTH_FILE", tmp_path / "google_auth.json")
        pipeline = chat_mod.build_pipeline("deepseek", None, None)
        monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")

        srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            root = f"http://127.0.0.1:{srv.server_address[1]}"
            st = json.loads(urllib.request.urlopen(root + "/auth/status", timeout=10).read().decode())
            assert st == {"signed_in": False, "configured": False}
            try:
                urllib.request.urlopen(root + "/auth/google", timeout=10)
                raise AssertionError("expected /auth/google to 400 without credentials")
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
                assert "not configured" in exc.read().decode().lower()
            req = urllib.request.Request(root + "/auth/signout", method="POST")
            d = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
            assert d["ok"] is True
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()


def test_stats_endpoint(tmp_path, monkeypatch):
    """GET /stats returns per-strategy aggregates + trend for the Insights panel."""
    import urllib.parse
    import urllib.request

    from dse import ask as ask_mod
    from dse import chat as chat_mod

    server, captured = _scripted_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("DSE_PROVIDER_URL", base)
        pipeline = chat_mod.build_pipeline("deepseek", None, None)
        monkeypatch.setattr(ask_mod, "SESSIONS_DIR", tmp_path / "sessions")

        srv = chat_mod.ChatServer(("127.0.0.1", 0), chat_mod.ChatHandler, pipeline, threading.Lock())
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            port = srv.server_address[1]
            # empty first
            stats = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=10).read().decode("utf-8"))
            assert stats["n"] == 0
            # add records, then stats has per-strategy aggregates
            url = f"http://127.0.0.1:{port}/ask?question=" + urllib.parse.quote("what is 2+2?")
            urllib.request.urlopen(url, timeout=30).read()
            stats = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=10).read().decode("utf-8"))
            assert stats["n"] == 4
            assert {s["strategy"] for s in stats["strategies"]} == {"react", "best_of_n", "reflexion", "self_refine"}
            assert all(s["judge_avg"] == 7.0 for s in stats["strategies"])
            assert all(s["n"] == 1 for s in stats["strategies"])
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        server.shutdown()
