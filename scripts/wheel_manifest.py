"""Print a JSON manifest of the built wheel(s): file name, sha256, the CPU baseline (MARCH) the native library was built for, the compiler, the commit.  usage: wheel_manifest.py <dist dir> <march>"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    dist, march = Path(sys.argv[1]), sys.argv[2]
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        print("no wheel in", dist, file=sys.stderr)
        return 1
    print(json.dumps({
        "commit": os.environ.get("GITHUB_SHA") or _run("git", "rev-parse", "HEAD"),
        "march": march,
        "compiler": (_run("g++", "--version").splitlines() or [""])[0],
        "wheels": [{"file": w.name, "sha256": hashlib.sha256(w.read_bytes()).hexdigest()} for w in wheels],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
