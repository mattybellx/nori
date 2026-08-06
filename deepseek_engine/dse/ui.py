"""Tiny terminal UI helpers — colors, sections, tables, progress bars.

Zero dependencies. Auto-disables color and overdraw when stdout is not a TTY
(pipes, logs, CI) so nothing ever garbles, and swaps Unicode glyphs for ASCII
when not on a real terminal (Windows legacy consoles / cp1252 pipes would
otherwise mangle them). On Windows, VT processing is enabled once so modern
consoles render ANSI correctly.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

_RESET = "\x1b[0m"
_COLOR: bool | None = None
_UNICODE: bool | None = None


def _enable_utf8() -> None:
    """Windows consoles often default to cp1252, which cannot encode the box
    drawing / symbol characters used by this UI. Reconfigure the streams to
    UTF-8 with lossy replacement so fancy output never crashes anywhere."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, AttributeError):
            pass


_enable_utf8()


def _color_enabled() -> bool:
    global _COLOR
    if _COLOR is None:
        _COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        if _COLOR and os.name == "nt":
            os.system("")  # enable VT processing on legacy Windows consoles
    return _COLOR


def _use_unicode() -> bool:
    global _UNICODE
    if _UNICODE is None:
        # Pretty glyphs only on a real terminal; ASCII everywhere else
        # (pipes, logs, legacy consoles) so output is never mojibaked.
        _UNICODE = sys.stdout.isatty() and not os.environ.get("NO_UNICODE")
    return _UNICODE


class _Glyphs:
    def __init__(self, unicode_ok: bool) -> None:
        if unicode_ok:
            self.bar = "━"
            self.hline = "─"
            self.fill = "█"
            self.empty = "░"
            self.ok = "✓"
            self.fail = "✗"
            self.warn = "⚠"
            self.ellipsis = "…"
            self.up = "▲"
            self.down = "▼"
            self.dot = "•"
        else:
            self.bar = "="
            self.hline = "-"
            self.fill = "#"
            self.empty = "."
            self.ok = "OK"
            self.fail = "X"
            self.warn = "!"
            self.ellipsis = "..."
            self.up = "+"
            self.down = "-"
            self.dot = "."


G = _Glyphs(_use_unicode())


def _c(code: str):
    def style(text: str) -> str:
        return f"\x1b[{code}m{text}{_RESET}" if _color_enabled() else text

    style._codes = code  # type: ignore[attr-defined]
    return style


def compose(*styles):
    """Combine styles (bold + cyan + ...) into one ANSI sequence."""
    codes = "".join(getattr(st, "_codes", "") for st in styles)
    if not codes:
        return lambda text: text

    def styled(text: str) -> str:
        return f"\x1b[{codes}m{text}{_RESET}" if _color_enabled() else text

    return styled


# -- styles -----------------------------------------------------------------
bold = _c("1")
dim = _c("2")
underline = _c("4")
red = _c("31")
green = _c("32")
yellow = _c("33")
blue = _c("34")
magenta = _c("35")
cyan = _c("36")
white = _c("37")
bright_red = _c("91")
bright_green = _c("92")
bright_yellow = _c("93")
bright_cyan = _c("96")


def _sanitize(text: str) -> str:
    """Replace non-ASCII punctuation in message bodies when not on a real
    terminal (so piped output is never mojibaked)."""
    if _use_unicode():
        return text
    return text.replace("—", "-").replace("→", "->").replace("…", "...")


def ok(text: str) -> str:
    return green(G.ok + " " + _sanitize(text))


def fail(text: str) -> str:
    return red(G.fail + " " + _sanitize(text))


def warn(text: str) -> str:
    return yellow(G.warn + " " + _sanitize(text))


def _term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except (OSError, ValueError):
        return 80


def section(title: str, color=cyan, char: str | None = None) -> None:
    """A bold, boxed section header, sized to the terminal."""
    char = char or G.bar
    if not _use_unicode():
        title = title.replace("—", "-").replace("→", "->")
    width = max(40, min(_term_width(), 100))
    bar = char * width
    print(color(bar))
    print(color(f"  {title.upper()}"))
    print(color(bar))


def hline(char: str | None = None, color=dim) -> None:
    char = char or G.hline
    print(color(char * max(40, min(_term_width(), 100))))


def _fmt_cell(value, width: int, align: str, style) -> str:
    text = f"{str(value):{align}{width}}"
    return style(text) if style else text


def table(
    headers: list[str],
    rows: list[list],
    *,
    header_style=None,
    cell_styles: list[list] | None = None,
    col_align: dict[int, str] | None = None,
    pad: int = 2,
) -> str:
    """Align columns on the *plain* text, then style cells (so color codes
    never break alignment). Returns the rendered block."""
    cell_styles = cell_styles or [[None] * len(headers) for _ in rows]
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    col_align = col_align or {}
    sep = " " * pad

    def render(values, styles) -> str:
        cells = [
            _fmt_cell(v, widths[i], col_align.get(i, "<"), styles[i])
            for i, v in enumerate(values)
        ]
        return sep.join(cells)

    lines = [render(headers, [header_style] * len(headers) if header_style else [None] * len(headers))]
    lines.append(G.hline * (sum(widths) + pad * (len(headers) - 1)))
    lines.extend(render(row, style) for row, style in zip(rows, cell_styles))
    return "\n".join(lines)


def progress(total: int, desc: str = "", width: int = 28):
    """Return a ``(update, finish)`` pair for a terminal progress bar.

    ``update(done, message="")`` redraws the bar (or stays silent when not a
    TTY, printing only a final line on ``finish``). ``finish(message="")``
    clears the bar and prints the completion line.
    """
    start = time.perf_counter()
    last_len = [0]
    is_tty = sys.stdout.isatty()

    def update(done: int, message: str = "") -> None:
        if not is_tty:
            return
        pct = (done / total) if total else 1.0
        filled = int(width * pct)
        bar = G.fill * filled + G.empty * (width - filled)
        elapsed = time.perf_counter() - start
        eta = (elapsed / pct - elapsed) if pct > 0 else 0.0
        line = (
            f"\r{desc:<12} [{bar}] {done:>3}/{total:<3} {pct * 100:5.1f}%  "
            f"{elapsed:6.1f}s  eta {eta:6.1f}s  {message}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        last_len[0] = len(line)

    def finish(message: str = "") -> None:
        elapsed = time.perf_counter() - start
        if not is_tty:
            print(f"{desc}: {total} done in {elapsed:.1f}s {message}".rstrip())
            return
        sys.stdout.write("\r" + " " * last_len[0] + "\r")
        print(f"{desc}: {total} done in {elapsed:.1f}s {message}".rstrip())

    return update, finish


def user_input(prompt_text: str) -> str:
    """Prompt wrapper that works in interactive and non-TTY contexts."""
    try:
        return input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def panel(title: str, body: str, title_color=cyan, body_color=None) -> str:
    """A lightweight labeled block for answers/records."""
    lines = [title_color(f"{G.hline} {title} " + G.hline * max(10, 40 - len(title)))]
    for line in body.splitlines():
        lines.append(f"  {line}" if body_color is None else body_color(f"  {line}"))
    return "\n".join(lines)
