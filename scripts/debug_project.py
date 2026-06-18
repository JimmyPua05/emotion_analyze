"""Run project diagnostics from the terminal.

Usage:
    python scripts/debug_project.py

Exit code:
    0 = no failed checks
    1 = at least one failed check
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.debug_tools import FAIL, format_diagnostics, run_diagnostics, summarize_diagnostics


def main() -> int:
    """Print diagnostics and return a useful terminal exit code."""

    checks = run_diagnostics(PROJECT_ROOT)
    summary = summarize_diagnostics(checks)
    print("Social Media Emotion Analyzer - Debug Report")
    print("=" * 52)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"PASS: {summary.get('PASS', 0)} | WARN: {summary.get('WARN', 0)} | FAIL: {summary.get('FAIL', 0)}")
    print("-" * 52)
    print(format_diagnostics(checks))
    return 1 if any(check["status"] == FAIL for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
