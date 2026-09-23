"""Module entry point: ``python3 -m sequence_engine --selftest``."""

import sys

from .selftest import run_selftest


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        ok, report = run_selftest()
        print(report)
        return 0 if ok else 1
    print("usage: python3 -m sequence_engine --selftest", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
