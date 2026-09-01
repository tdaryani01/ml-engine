"""Compare controlled pre-A vs post-trim A/B logs."""
import statistics as stats
from pathlib import Path

from compare_phase_logs import parse_summary, pct, DIAG

PRE = DIAG / "conv_ab_pre-a_k1-3-4-7_s1-2_p1-2_20260901_044724.log"
POST = DIAG / "conv_ab_post-trim_k1-3-4-7_s1-2_p1-2_20260901_045740.log"

pre = parse_summary(PRE)
post = parse_summary(POST)
keys = sorted(set(pre) & set(post))

print("Controlled A/B (same harness, rebuild each side)")
print(f"  pre-A:     {PRE.name}")
print(f"  post-trim: {POST.name}")
print(f"  cases:     {len(keys)}\n")

print(f"{'k s p':>7} | {'pre-A':>10} | {'post-trim':>10} | {'delta':>8}")
print("-" * 45)
deltas = []
for k in keys:
    a, b = pre[k], post[k]
    d = pct(b, a)
    deltas.append(d)
    print(f"{k[0]:2d} {k[1]} {k[2]} | {a:10.2f} | {b:10.2f} | {d:+7.1f}%")

print(
    f"\npost-trim vs pre-A: median {stats.median(deltas):+.1f}%, "
    f"mean {stats.mean(deltas):+.1f}%, max {max(deltas):+.1f}%, min {min(deltas):+.1f}%"
)
within = sum(1 for d in deltas if abs(d) <= 3)
print(f"Within ±3%: {within}/{len(keys)} cases")
