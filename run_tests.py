import sys
import subprocess
import time


def run_test_module(name: str, script_path: str) -> bool:
    print(f"\n{'='*70}")
    print(f" [RUNNING] {name} ({script_path})")
    print(f"{'='*70}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"--> {name}: PASSED ({elapsed:.2f}s)")
        return True
    else:
        print(f"--> {name}: FAILED ({elapsed:.2f}s)")
        return False


def main():
    test_suite = [
        ("Tier 1: Optimizers", "testing/test_optimizers.py"),
        ("Tier 1: LR Schedulers", "testing/test_schedulers.py"),
        ("Tier 1: Serializer State", "testing/test_serializer.py"),
        ("Tier 2: Autodiff Gradient Check", "testing/test_gradient_check.py"),
        ("Tier 3: Pipeline Integration", "testing/test_pipeline_integration.py"),
    ]

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
        print(f" - {name:<35} : {status}")

    print("-" * 70)
    print(f"Total Duration: {total_elapsed:.2f}s")
    if all_passed:
        print("\n[READY TO COMMIT] All unit, gradient, and pipeline tests passed!\n")
        sys.exit(0)
    else:
        print("\n[COMMIT BLOCKED] One or more tests failed. Review output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()