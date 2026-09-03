# run_tests.py
"""Run the full local test suite (all testing/test_*.py modules)."""
import glob
import os
import re
import subprocess
import sys
import time

LOG_FILE = "test_run.log"

# Preferred tier order; any new test_*.py not listed here runs at the end.
_TIER_ORDER = [
    "testing/test_config.py",
    "testing/test_optimizers.py",
    "testing/test_schedulers.py",
    "testing/test_serializer.py",
    "testing/test_im2col.py",
    "testing/test_engine_ops.py",
    "testing/test_runtime.py",
    "testing/test_training_cache.py",
    "testing/test_training_session.py",
    "testing/test_ledger.py",
    "testing/test_im2col_gemm.py",
    "testing/test_native_conv.py",
    "testing/test_benchmark_harness.py",
    "testing/test_gradient_check.py",
    "testing/test_spatial_layers.py",
    "testing/test_pipeline_integration.py",
    "testing/test_cnn_pipeline.py",
]


class DualWriter:
    """Duplicate stdout/stderr to console and a log file."""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            enc = self.terminal.encoding or "utf-8"
            safe = message.encode(enc, errors="replace").decode(enc, errors="replace")
            self.terminal.write(safe)
        if not self.log.closed:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()


def discover_test_modules(project_root: str) -> list[tuple[str, str]]:
    """Return (label, path) for every testing/test_*.py module."""
    pattern = os.path.join(project_root, "testing", "test_*.py")
    discovered = sorted(glob.glob(pattern))
    order_index = {path.replace("\\", "/"): idx for idx, path in enumerate(_TIER_ORDER)}

    def sort_key(path: str) -> tuple:
        normalized = os.path.relpath(path, project_root).replace("\\", "/")
        tier = order_index.get(normalized, len(_TIER_ORDER))
        return (tier, normalized)

    modules = []
    for path in sorted(discovered, key=sort_key):
        rel = os.path.relpath(path, project_root).replace("\\", "/")
        label = rel.replace("testing/test_", "").replace(".py", "").replace("_", " ").title()
        modules.append((label, rel))
    return modules


def parse_sub_backend_results(stdout_text: str) -> list:
    """Extract explicit sub-backend execution statuses from test output."""
    sub_results = []

    backend_blocks = re.split(r"--- (?:Testing )?Backend: ([a-zA-Z0-9_\+\-]+) ---", stdout_text)
    if len(backend_blocks) > 1:
        for i in range(1, len(backend_blocks), 2):
            b_name = backend_blocks[i].strip()
            block_content = backend_blocks[i + 1] if i + 1 < len(backend_blocks) else ""
            b_passed = not ("FAILED" in block_content or "ERROR" in block_content or "Traceback" in block_content)
            sub_results.append((b_name, b_passed))
        return sub_results

    grad_matches = re.findall(
        r"RUNNING CHECK: Task='([^']+)'(?:\s*\|\s*Backend='([^']+)')?", stdout_text
    )
    if grad_matches:
        for task, b_name in grad_matches:
            label = f"{task} ({b_name})" if b_name else task
            passed = f"Discrepancy detected in {task}" not in stdout_text and "Traceback" not in stdout_text
            sub_results.append((label, passed))
        return sub_results

    return sub_results


def run_test_module(name: str, script_path: str) -> tuple[bool, list]:
    print(f"\n{'=' * 70}")
    print(f" [RUNNING] {name} ({script_path})")
    print(f"{'=' * 70}")

    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else project_root
    )
    env["PYTHONIOENCODING"] = "utf-8"

    start_time = time.time()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins; builtins.profile=getattr(builtins,'profile',lambda f:f); "
            "import runpy; runpy.run_path(%r, run_name='__main__')" % script_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=project_root,
    )
    elapsed = time.time() - start_time

    if result.stdout:
        print(result.stdout, end="")

    sub_backends = parse_sub_backend_results(result.stdout or "")
    passed = result.returncode == 0

    if passed:
        print(f"--> {name}: PASSED ({elapsed:.2f}s)")
    else:
        print(f"--> {name}: FAILED (exit {result.returncode}, {elapsed:.2f}s)")

    return passed, sub_backends


def main():
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    writer = DualWriter(LOG_FILE)
    sys.stdout = writer
    sys.stderr = writer

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_suite = discover_test_modules(project_root)
    exit_code = 0

    try:
        print("\n" + "#" * 70)
        print("           ML-ENGINE FULL LOCAL TEST HARNESS")
        print("#" * 70)
        print(f"Discovered {len(test_suite)} test module(s) in testing/")

        total_start = time.time()
        all_passed = True
        summary = []

        for name, path in test_suite:
            passed, sub_backends = run_test_module(name, path)
            summary.append((name, path, passed, sub_backends))
            if not passed:
                all_passed = False

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print("                      FULL TEST SUITE SUMMARY")
        print("=" * 70)

        for name, path, passed, sub_backends in summary:
            status = "PASSED [OK]" if passed else "FAILED [ERR]"
            print(f" - {name:<40} ({path}) : {status}")
            if sub_backends:
                for idx, (sub_name, sub_passed) in enumerate(sub_backends):
                    branch = "└──" if idx == len(sub_backends) - 1 else "├──"
                    sub_status = "PASSED [OK]" if sub_passed else "FAILED [ERR]"
                    print(f"     {branch} {sub_name:<42} : {sub_status}")

        print("\n" + "-" * 70)
        print(f"Total Duration: {total_elapsed:.2f}s")
        if all_passed:
            print(
                f"\n[READY TO COMMIT] All {len(test_suite)} test modules passed. "
                f"Log saved to: {os.path.abspath(LOG_FILE)}\n"
            )
            exit_code = 0
        else:
            failed = sum(1 for _, _, ok, _ in summary if not ok)
            print(
                f"\n[COMMIT BLOCKED] {failed}/{len(test_suite)} test module(s) failed. "
                f"Full log: {os.path.abspath(LOG_FILE)}\n"
            )
            exit_code = 1

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        writer.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
