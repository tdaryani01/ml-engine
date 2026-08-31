# run_tests.py
import sys
import subprocess
import time
import os
import re

LOG_FILE = "test_run.log"


class DualWriter:
    """
    Intercepts standard output/error and duplicates every print/stream
    call to both the console terminal and a persistent log file on disk.
    """
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            self.terminal.write(message.encode(self.terminal.encoding or "utf-8", errors="replace").decode(self.terminal.encoding or "utf-8", errors="replace"))
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


def parse_sub_backend_results(stdout_text: str) -> list:
    """Extracts explicit sub-backend execution statuses from test output."""
    sub_results = []
    
    # 1. Match standard backend sections: --- Backend: <name> --- or --- Testing Backend: <name> ---
    backend_blocks = re.split(r"--- (?:Testing )?Backend: ([a-zA-Z0-9_\+\-]+) ---", stdout_text)
    if len(backend_blocks) > 1:
        for i in range(1, len(backend_blocks), 2):
            b_name = backend_blocks[i].strip()
            block_content = backend_blocks[i+1] if i+1 < len(backend_blocks) else ""
            b_passed = not ("FAILED" in block_content or "ERROR" in block_content or "Traceback" in block_content)
            sub_results.append((b_name, b_passed))
        return sub_results

    # 2. Match Task & Backend lines (test_gradient_check style)
    grad_matches = re.findall(r"RUNNING CHECK: Task='([^']+)'(?:\s*\|\s*Backend='([^']+)')?", stdout_text)
    if grad_matches:
        for task, b_name in grad_matches:
            label = f"{task} ({b_name})" if b_name else task
            # Check if this section had a failure
            passed = f"Discrepancy detected in {task}" not in stdout_text and "Traceback" not in stdout_text
            sub_results.append((label, passed))
        return sub_results

    return sub_results


def run_test_module(name: str, script_path: str, backend: str) -> tuple:
    backend_label = f"BACKEND={backend.upper()}"
    print(f"\n{'='*70}")
    print(f" [RUNNING] [{backend_label}] {name} ({script_path})")
    print(f"{'='*70}")
    
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else project_root
    env["PYTHONIOENCODING"] = "utf-8"
    env["ENGINE_BACKEND"] = backend
    
    start_time = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=project_root
    )
    elapsed = time.time() - start_time
    
    if result.stdout:
        print(result.stdout, end="")
    
    sub_backends = parse_sub_backend_results(result.stdout or "")
    passed = (result.returncode == 0)
    
    if passed:
        print(f"--> [{backend.upper()}] {name}: PASSED ({elapsed:.2f}s)")
    else:
        print(f"--> [{backend.upper()}] {name}: FAILED ({elapsed:.2f}s)")
        
    return passed, sub_backends


def main():
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    writer = DualWriter(LOG_FILE)
    sys.stdout = writer
    sys.stderr = writer

    test_suite = [
        # --- Tier 1: Core Primitives & Utilities ---
        ("Tier 1: Optimizers", "testing/test_optimizers.py"),
        ("Tier 1: LR Schedulers", "testing/test_schedulers.py"),
        ("Tier 1: Serializer State", "testing/test_serializer.py"),
        ("Tier 1: im2col / col2im Ops", "testing/test_im2col.py"),

        # --- Tier 2: Layers & Gradient Verification ---
        ("Tier 2: Native Conv Parity & Dispatch", "testing/test_native_conv.py"),
        ("Tier 2: Benchmark Harness Regression", "testing/test_benchmark_harness.py"),
        ("Tier 2: Autodiff Gradient Check", "testing/test_gradient_check.py"),
        ("Tier 2: Spatial Layers & Conv2D Grad", "testing/test_spatial_layers.py"),

        # --- Tier 3: End-to-End Pipelines ---
        ("Tier 3: Pipeline Integration (MLP)", "testing/test_pipeline_integration.py"),
        ("Tier 3: CNN Integration Pipeline", "testing/test_cnn_pipeline.py"),
    ]

    target_backends = ["fast", "numpy"]
    exit_code = 0

    try:
        print("\n" + "#" * 70)
        print("      ML-ENGINE LOCAL DUAL-BACKEND PRE-COMMIT TEST HARNESS")
        print("#" * 70)

        total_start = time.time()
        all_passed = True
        summary = []

        for backend in target_backends:
            print(f"\n{'#'*70}")
            print(f" >>> EXECUTING SUITE MATRIX: ENGINE_BACKEND = '{backend}' <<<")
            print(f"{'#'*70}")
            
            for name, path in test_suite:
                passed, sub_backends = run_test_module(name, path, backend)
                summary.append((backend.upper(), name, passed, sub_backends))
                if not passed:
                    all_passed = False

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print("                    DUAL-BACKEND TEST SUITE SUMMARY")
        print("=" * 70)

        current_backend_header = None
        for backend_tag, name, passed, sub_backends in summary:
            if current_backend_header != backend_tag:
                current_backend_header = backend_tag
                print(f"\n[ENGINE MATRIX: {current_backend_header}]")

            status = "PASSED [OK]" if passed else "FAILED [ERR]"
            print(f" - {name:<46} : {status}")
            
            if sub_backends:
                for idx, (sub_name, sub_passed) in enumerate(sub_backends):
                    is_last = (idx == len(sub_backends) - 1)
                    branch = "└──" if is_last else "├──"
                    sub_status = "PASSED [OK]" if sub_passed else "FAILED [ERR]"
                    print(f"     {branch} {sub_name:<42} : {sub_status}")

        print("\n" + "-" * 70)
        print(f"Total Combined Duration: {total_elapsed:.2f}s")
        if all_passed:
            print(f"\n[READY TO COMMIT] All dual-matrix tests passed! Log saved to: {os.path.abspath(LOG_FILE)}\n")
            exit_code = 0
        else:
            print(f"\n[COMMIT BLOCKED] One or more tests failed. Full log: {os.path.abspath(LOG_FILE)}\n")
            exit_code = 1

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        writer.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()