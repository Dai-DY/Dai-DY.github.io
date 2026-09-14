#!/usr/bin/env python3

import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
COLORIZE_SCRIPT = PROJECT_DIR / "colorize_skel.py"


def main() -> None:
    source = COLORIZE_SCRIPT.read_text(encoding="utf-8")
    input_pattern = re.compile(r"^imname\s*=.*$", re.MULTILINE)

    if not input_pattern.search(source):
        raise RuntimeError("找不到 colorize_skel.py 中的 imname 设置")

    input_files = sorted(path for path in DATA_DIR.rglob("*") if path.is_file())
    (PROJECT_DIR / "output").mkdir(exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    for input_file in input_files:
        relative_path = input_file.relative_to(PROJECT_DIR)
        print(f"Colorizing {relative_path}", flush=True)

        file_source = input_pattern.sub(
            f"imname = {str(relative_path)!r}", source, count=1
        )
        subprocess.run(
            [sys.executable, "-"],
            cwd=PROJECT_DIR,
            env=env,
            input=file_source,
            text=True,
            check=True,
        )


if __name__ == "__main__":
    main()
