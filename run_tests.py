# run_tests.py
import sys
import subprocess
import time
import os

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
        self.terminal.write(message)
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


def run_test_module(name: str, script_path: str) -> bool:
    print(f"\n{'='*70}")
    print(f" [RUNNING] {name} ({script_path})")
    print(f"{'='*70}")
    
    # Configure child process environment with root in PYTHONPATH and UTF-8 encoding
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else project_root
    env["PYTHONIOENCODING"] = "utf-8"
    
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
    
    # Stream child process stdout and tracebacks
    if result.stdout:
        print(result.stdout, end="")
    
    if result.returncode == 0:
        print(f"--> {name}: PASSED ({elapsed:.2f}s)")
        return True
    else:
        print(f"--> {name}: FAILED ({elapsed:.2f}s)")
        return False


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
        ("Tier 2: Autodiff Gradient Check", "testing/test_gradient_check.py"),
        ("Tier 2: Spatial Layers & Conv2D Grad", "testing/test_spatial_layers.py"),

        # --- Tier 3: End-to-End Pipelines ---
        ("Tier 3: Pipeline Integration (MLP)", "testing/test_pipeline_integration.py"),
        ("Tier 3: CNN Integration Pipeline", "testing/test_cnn_pipeline.py"),
    ]

    exit_code = 0
    try:
        print("\n" + "#" * 70)
        print("      ML-ENGINE LOCAL PRE-COMMIT TEST HARNESS")
        print("#" * 70)

        total_start = time.time()
        all_passed = True
        summary = []

        for name, path in test_suite:
            passed = run_test_module(name, path)
            summary.append((name, passed))
            if not passed:
                all_passed = False

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print("                    TEST SUITE SUMMARY")
        print("=" * 70)
        for name, passed in summary:
            status = "PASSED [OK]" if passed else "FAILED [ERR]"
            print(f" - {name:<40} : {status}")

        print("-" * 70)
        print(f"Total Duration: {total_elapsed:.2f}s")
        if all_passed:
            print(f"\n[READY TO COMMIT] All tests passed! Log saved to: {os.path.abspath(LOG_FILE)}\n")
            exit_code = 0
        else:
            print(f"\n[COMMIT BLOCKED] One or more tests failed. Full log: {os.path.abspath(LOG_FILE)}\n")
            exit_code = 1

    finally:
        # Restore native stdout/stderr handles before closing writer to avoid unraisable hook exceptions
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        writer.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()