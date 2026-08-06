"""Scripted OpenAI-compatible server for smoke-testing the chat UI."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8799


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        system = body.get("messages", [{}])[0].get("content", "")
        if "MODE: grade" in system:
            content = "SCORE: 7/10\nFEEDBACK: solid answer"
        elif "Pick the single best answer" in system:
            content = "BEST: 2\nREASON: the second answer is the most accurate, clear and complete\nWEAKNESS: it could add a short example"
        else:
            content = (
                "The sky looks blue because of Rayleigh scattering — shorter "
                "wavelengths of sunlight are scattered more strongly by the "
                "atmosphere, so we see blue from every direction."
            )
        payload = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 18},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"scripted provider on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
