"""Fixed sample sweep (k=1,3,4,7 x stride 1,2 x pad 1,2) for controlled A/B runs."""
import os
import sys
from datetime import datetime, timezone

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from benchmarks.sweep_kernel_pad import run_sweep  # noqa: E402

KERNELS = [1, 3, 4, 7]
STRIDES = [1, 2]
PADS = [1, 2]


def main() -> None:
    tag = os.environ.get("AB_TAG", "ab")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    diag = os.path.join(project_root, "benchmark_diagnostics")
    os.makedirs(diag, exist_ok=True)
    out = os.environ.get(
        "AB_OUTPUT",
        os.path.join(diag, f"conv_ab_{tag}_k1-3-4-7_s1-2_p1-2_{stamp}.log"),
    )
    print(f"[AB sample] tag={tag} output={out}", flush=True)
    run_sweep(
        kernel_sizes=KERNELS,
        pads=PADS,
        strides=STRIDES,
        output_path=out,
    )
    print(f"[AB sample] complete: {out}", flush=True)


if __name__ == "__main__":
    main()
