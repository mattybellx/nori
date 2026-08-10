"""Google OAuth 2.0 sign-in + Google-Drive-backed chat-history sync.

stdlib-only (urllib). Flow:

  1. User clicks "Sign in with Google" -> server 302-redirects to Google
     consent (authorization-code flow, offline access so we get a refresh
     token).
  2. Google redirects to ``/auth/callback?code=...`` -> the server exchanges
     the code for access + refresh tokens and stores them in
     ``google_auth.json`` (local, git-ignored).
  3. The server pushes the local sessions folder to the app's private Drive
     **AppData** folder (invisible to the user's Drive, only the app can see
     it). ``/auth/sync`` also pulls the remote copy back and merges it into
     the local sessions, so chat history follows the user across machines.

To enable, create a Google Cloud OAuth 2.0 Client (Desktop / Web) and add
``google_client_id`` / ``google_client_secret`` to ``settings.json`` (or set
``DSE_GOOGLE_CLIENT_ID`` / ``DSE_GOOGLE_CLIENT_SECRET`` in the environment).
Register the exact redirect URI ``http://127.0.0.1:<port>/auth/callback`` in
the Google Cloud Console under Authorized redirect URIs. Without credentials
the sign-in button is disabled and the API returns a clear 400.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
SCOPE = "https://www.googleapis.com/auth/drive.appdata openid email"
APP_FILE = "nori_history.json"

#: Local token store (git-ignored).
AUTH_FILE = Path(__file__).resolve().parent.parent / "google_auth.json"


# ---------------------------------------------------------------------------
# Credentials + token storage
# ---------------------------------------------------------------------------
def client_credentials(settings: dict) -> tuple[str, str]:
    cid = settings.get("google_client_id") or os.environ.get("DSE_GOOGLE_CLIENT_ID", "")
    secret = settings.get("google_client_secret") or os.environ.get(
        "DSE_GOOGLE_CLIENT_SECRET", "")
    return cid, secret


def configured(settings: dict) -> bool:
    return bool(client_credentials(settings)[0])


def load_auth() -> dict | None:
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data or None


def save_auth(data: dict) -> None:
    try:
        AUTH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort


def clear_auth() -> None:
    try:
        AUTH_FILE.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# OAuth endpoints
# ---------------------------------------------------------------------------
def build_auth_url(client_id: str, redirect_uri: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def _post_form(url: str, fields: dict) -> dict:
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def exchange_code(code: str, client_id: str, client_secret: str,
                  redirect_uri: str) -> dict:
    """Trade the authorization code for tokens (raises on failure)."""
    return _post_form(TOKEN_URL, {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    })


def refresh_access(refresh_token: str, client_id: str, client_secret: str) -> dict:
    return _post_form(TOKEN_URL, {
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
    })


def ensure_token(auth: dict, settings: dict) -> dict:
    """Return a fresh access token, refreshing when expired."""
    if auth.get("access_token") and time.time() < auth.get("expires_at", 0):
        return auth
    refresh = auth.get("refresh_token")
    if not refresh:
        return auth
    cid, secret = client_credentials(settings)
    tok = refresh_access(refresh, cid, secret)
    auth.update({**tok, "expires_at": time.time() + int(tok.get("expires_in", 3600)) - 60})
    save_auth(auth)
    return auth


# ---------------------------------------------------------------------------
# Google APIs
# ---------------------------------------------------------------------------
def _api(method: str, url: str, token: str, body: bytes | None = None,
         content_type: str = "application/json") -> tuple[int, dict | str]:
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if body is not None:
        data = body
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def whoami(token: str) -> dict:
    """Signed-in user's profile (email, name) — best-effort."""
    try:
        _status, body = _api("GET", USERINFO_URL, token)
        if isinstance(body, dict):
            return body
    except Exception:
        pass
    return {}


def _drive_file_id(token: str) -> str | None:
    """Find our history file inside the app's private Drive AppData folder."""
    if not token:
        return None
    url = DRIVE_FILES_URL + "?spaces=appDataFolder&fields=files(id,name)"
    _status, body = _api("GET", url, token)
    if isinstance(body, dict):
        for f in body.get("files", []):
            if f.get("name") == APP_FILE:
                return f.get("id")
    return None


def _multipart(metadata: dict, media: bytes,
               media_type: str = "application/json") -> tuple[bytes, str]:
    boundary = "----nori" + str(time.time_ns())
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n"
        f"Content-Type: application/json\r\n\r\n{json.dumps(metadata)}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"media\"\r\n"
        f"Content-Type: {media_type}\r\n\r\n".encode() + media + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/related; boundary={boundary}"


# ---------------------------------------------------------------------------
# History sync (bundle <-> Drive AppData)
# ---------------------------------------------------------------------------
def bundle_sessions(sessions_dir) -> dict:
    """Flatten every sessions/*.jsonl record into one portable bundle."""
    records: list[dict] = []
    for path in sorted(Path(sessions_dir).glob("ask_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return {
        "app": "nori",
        "version": 1,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
    }


def sync_push(token: str, sessions_dir) -> dict:
    """Upload the local sessions to Drive AppData (create or update)."""
    media = json.dumps(bundle_sessions(sessions_dir),
                       ensure_ascii=False).encode("utf-8")
    file_id = _drive_file_id(token)
    if file_id:
        _status, body = _api("PATCH", f"{DRIVE_UPLOAD_URL}/{file_id}?uploadType=media",
                             token, body=media, content_type="application/json")
    else:
        body_bytes, ctype = _multipart({"name": APP_FILE, "parents": ["appDataFolder"]}, media)
        _status, body = _api("POST", f"{DRIVE_UPLOAD_URL}?uploadType=multipart",
                             token, body=body_bytes, content_type=ctype)
    return {"ok": True, "file_id": file_id, "body": body}


def sync_pull(token: str) -> dict:
    """Download the remote bundle (empty records when none exists yet)."""
    file_id = _drive_file_id(token)
    if not file_id:
        return {"ok": True, "records": [], "file_id": None}
    _status, body = _api("GET", f"{DRIVE_FILES_URL}/{file_id}?alt=media", token)
    if isinstance(body, dict) and isinstance(body.get("records"), list):
        return {"ok": True, "records": body["records"], "file_id": file_id}
    return {"ok": True, "records": [], "file_id": file_id}


def merge_pull(remote_records: list[dict], local_records: list[dict]) -> list[dict]:
    """Records from the cloud that aren't already stored locally (append-only
    by question+strategy+ts)."""
    local_keys = {(r.get("question"), r.get("strategy"), r.get("ts"))
                  for r in local_records}
    return [r for r in remote_records
            if (r.get("question"), r.get("strategy"), r.get("ts")) not in local_keys]
