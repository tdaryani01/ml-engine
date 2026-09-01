"""Compare custom ms/epoch across base, Phase A, and Phase B sample sweeps."""
import re
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "benchmark_diagnostics"

LOGS = {
    "base (matrix pre-A)": DIAG / "conv_matrix_k1-7_s1-2_p1-2_20260901_023244.log",
    "phase A (sample)": DIAG / "conv_sample_k1-3-4-7_s1-2_p1-2_20260901_035604.log",
    "phase B (sample)": DIAG / "conv_sample_k1-3-4-7_s1-2_p1-2_20260901_041222.log",
    "hot-path trim": DIAG / "conv_sample_k1-3-4-7_s1-2_p1-2_20260901_043346.log",
}


def parse_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = {}
    in_summary = False
    for line in text.splitlines():
        if "SWEEP SUMMARY" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.strip().startswith("Ratio =") or line.strip().startswith("Completed "):
            break
        m = re.match(
            r"\s*(\d+)\s+(\d+)\s+(\d+)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+x\s+\|\s+[\d.]+\s+\|\s+([\d.]+)",
            line,
        )
        if m:
            k, s, p = map(int, m.group(1, 2, 3))
            rows[(k, s, p)] = float(m.group(4))
    return rows


def pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def main() -> None:
    parsed = {name: parse_summary(p) for name, p in LOGS.items()}
    keys = sorted(set.intersection(*(set(d.keys()) for d in parsed.values())))
    base_key = "base (matrix pre-A)"
    trim_key = "hot-path trim"

    print("Logs compared:")
    for name, path in LOGS.items():
        print(f"  {name}: {path.name}")
    print(f"\nOverlapping cases (k in 1,3,4,7 x stride 1,2 x pad 1,2): {len(keys)}\n")

    print(f"{'k s p':>7} | {'base ms/ep':>11} | {'trim ms/ep':>11} | {'trim vs base':>12}")
    print("-" * 50)
    for key in keys:
        b = parsed[base_key][key]
        t = parsed[trim_key][key]
        k, s, p = key
        print(f"{k:2d} {s} {p} | {b:11.2f} | {t:11.2f} | {pct(t, b):+11.1f}%")

    tb = [pct(parsed[trim_key][k], parsed[base_key][k]) for k in keys]

    print("\nAggregate trim vs base (custom ms/epoch):")
    print(
        f"  median {stats.median(tb):+.1f}%, mean {stats.mean(tb):+.1f}%, "
        f"max {max(tb):+.1f}%, min {min(tb):+.1f}%"
    )

    threshold = 3.0
    within = sum(1 for v in tb if abs(v) <= threshold)
    print(f"\nRegression gate (within ±{threshold:.0f}% vs base): {within}/{len(keys)} cases")
    outliers = [(k, pct(parsed[trim_key][k], parsed[base_key][k])) for k in keys if abs(pct(parsed[trim_key][k], parsed[base_key][k])) > threshold]
    if outliers:
        print("  Outside gate:")
        for (k, s, p), d in sorted(outliers, key=lambda x: abs(x[1]), reverse=True):
            print(f"    k={k} s={s} p={p}: {d:+.1f}%")


if __name__ == "__main__":
    main()
