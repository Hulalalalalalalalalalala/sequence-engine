"""Entry point: python3 -m sequence_engine --selftest"""

from __future__ import annotations

import sys

from ._selftest import run_selftest

_USAGE = "usage: python3 -m sequence_engine --selftest"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--selftest"]:
        return run_selftest()
    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
