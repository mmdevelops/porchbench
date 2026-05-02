"""Console / stdout encoding helpers.

Kept as a tiny standalone module so both the CLI entry point and library
entry points (e.g. `evaluator.batch_evaluate_results`) can guard against
captured-output streams that default to a non-Unicode codepage.
"""

from __future__ import annotations

import sys


def ensure_unicode_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows when possible.

    Idempotent. Safe to call from any library entry point that may emit
    Rich output (spinners, box-drawing, em-dashes, sparklines). Without
    this, captured-output streams on Windows (pipes, file redirects, CI
    logs, harness subprocess stdout) default to cp1252 and crash on
    `UnicodeEncodeError` the moment Rich emits a non-Latin-1 glyph.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # pytest capfd / capsys replacement streams
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # already-detached or non-reconfigurable; fall through
