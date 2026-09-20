"""Runtime file-access audit wrapper (2026-08-15). No `strace` binary available
in this environment, so this uses Python's own `sys.addaudithook` on the
'open' event (covers io.open / numpy.load / nibabel.load / pickle -- anything
going through CPython's open()) to catch any access to vessel_masks/ during
M1 inference or mask building, without relying on manual code inspection.

Usage: audit_wrapper.py <target.py> [args...]
Runs target.py as __main__ with the remaining argv, printing
AUDIT_VIOLATION: <path> to stderr for every 'open' event whose path contains
"vessel_mask", and writing a summary line at exit.
"""
from __future__ import annotations

import sys

_violations: list[str] = []
_n_opens = 0


def _audit_hook(event: str, args: tuple) -> None:
    global _n_opens
    if event == "open":
        _n_opens += 1
        path = args[0]
        if isinstance(path, str) and "vessel_mask" in path:
            _violations.append(path)
            print(f"AUDIT_VIOLATION: opened {path}", file=sys.stderr, flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: audit_wrapper.py <target.py> [args...]", file=sys.stderr)
        return 2
    sys.addaudithook(_audit_hook)

    target = sys.argv[1]
    sys.argv = sys.argv[1:]  # target script sees itself as argv[0]

    with open(target) as f:
        code = compile(f.read(), target, "exec")
    g = {"__name__": "__main__", "__file__": target}
    try:
        exec(code, g)
    finally:
        print(f"\n[audit_wrapper] total open() calls observed: {_n_opens}", flush=True)
        print(f"[audit_wrapper] vessel_masks violations: {len(_violations)}", flush=True)
        if _violations:
            print(f"[audit_wrapper] violating paths: {_violations}", flush=True)
    return 1 if _violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
