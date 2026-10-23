#!/usr/bin/env python3
"""One PDF in, one reading-edition folder out.

    python pdf2reading.py paper.pdf -o out/paper
    python pdf2reading.py paper.pdf -o out/paper --load-4bit     # 8 GB card
    python pdf2reading.py paper.pdf -o out/paper --gold          # no model

The work is corpus/infer.py: the converter extracts the page's lines, the
model decides the reading structure, and reading_build/render.py draws it.
This file only resolves paths inside the checkout, so the command works from
any directory.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", required=True,
                    help="output folder (index.html and img/ go here)")
    ap.add_argument("--adapter", default=str(HERE / "adapter"),
                    help="LoRA adapter folder (default: ./adapter)")
    ap.add_argument("--gold", action="store_true",
                    help="skip the model and render the converter's own "
                         "structure; useful to tell a renderer bug from a "
                         "model bug")
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit base model, about 2.4 GB (8 GB cards); the "
                         "answer can differ from a bf16 run")
    ap.add_argument("--load-8bit", action="store_true",
                    help="8-bit base model, about 4.3 GB, closer to bf16")
    ap.add_argument("--max-new", type=int, default=2048,
                    help="generation budget per page")
    ap.add_argument("--dpi", type=int, default=24,
                    help="resolution of the structural pass; crops always "
                         "render at 200 dpi")
    ap.add_argument("--limit", type=int, default=0,
                    help="convert only the first N pages -- try the speed and "
                         "the look on a paper before running all of it")
    a = ap.parse_args()

    pdf = Path(a.pdf)
    if not pdf.is_file():
        sys.exit(f"no such file: {pdf}")

    cmd = [sys.executable, str(HERE / "corpus" / "infer.py"),
           "--pdf", str(pdf), "--out", str(a.out),
           "--max-new", str(a.max_new), "--dpi", str(a.dpi)]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    if a.gold:
        cmd.append("--gold")
    else:
        cmd += ["--adapter", a.adapter]
        if a.load_4bit:
            cmd.append("--load-4bit")
        if a.load_8bit:
            cmd.append("--load-8bit")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
