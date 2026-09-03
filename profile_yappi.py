# profile_yappi.py
"""Run the training pipeline under yappi (multi-thread) and save a snakeviz-ready profile."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile run_pipeline with yappi")
    parser.add_argument(
        "-o",
        "--output",
        default="yappi.pstat",
        help="Output path (pstat for snakeviz, or .callgrind)",
    )
    parser.add_argument(
        "--clock",
        choices=("wall", "cpu"),
        default="wall",
        help="yappi clock type (wall shows waits; cpu shows burn)",
    )
    parser.add_argument(
        "--snakeviz",
        action="store_true",
        help="Open snakeviz on the saved pstat when done",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also dump function/thread stats to the terminal (noisy)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="snakeviz bind host")
    parser.add_argument("--port", type=int, default=9002, help="snakeviz port")
    args = parser.parse_args()

    try:
        import yappi
    except ImportError:
        print("yappi not installed. Run: pip install yappi snakeviz", file=sys.stderr)
        return 1

    from run_pipeline import execute_training_pipeline

    yappi.set_clock_type(args.clock)
    yappi.clear_stats()
    yappi.start(builtins=False)
    try:
        execute_training_pipeline()
    finally:
        yappi.stop()

    stats = yappi.get_func_stats()
    out = args.output
    if out.endswith(".callgrind"):
        stats.save(out, type="callgrind")
    else:
        if not out.endswith(".pstat"):
            out = out if "." in out else f"{out}.pstat"
        stats.save(out, type="pstat")

    print(f"[yappi] saved {out} (clock={args.clock})")
    print(f"[yappi] view with: snakeviz {out}")

    if args.print:
        stats.sort("ttot")
        stats.print_all(
            out=sys.stdout,
            columns={
                0: ("name", 80),
                1: ("ncall", 10),
                2: ("tsub", 12),
                3: ("ttot", 12),
                4: ("tavg", 12),
            },
        )
        print("[yappi] threads:")
        for tstat in yappi.get_thread_stats():
            print(
                f"  id={tstat.id} name={tstat.name!r} tid={tstat.tid} "
                f"ttot={tstat.ttot:.4f}"
            )

    if args.snakeviz:
        if not out.endswith(".pstat"):
            print("[yappi] --snakeviz requires a .pstat output", file=sys.stderr)
            return 1
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "snakeviz",
                    "-H",
                    args.host,
                    "-p",
                    str(args.port),
                    out,
                ],
                check=False,
            )
        except FileNotFoundError:
            print("snakeviz not installed. Run: pip install snakeviz", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
