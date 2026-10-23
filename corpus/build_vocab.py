#!/usr/bin/env python3
"""Count words in a PDF sample to settle one question: is that hyphen a break?

A line ending in "-" is either a word broken by the typesetter ("Com- pared")
or a real compound broken at its hyphen ("signal- flow").  Joining both the
same way prints "Compaired" or "signalflow"; keeping both prints "Com-pared".
Neither is rare — there are about eighty such lines per paper.

The parts tell them apart: a hyphenation leaves a fragment that is not a word
("guaran", "func"), a compound leaves two words ("signal", "flow").  That
needs a vocabulary, which this script builds from the corpus itself — no
dictionary to install, and the domain words are already in it.

    python3 build_vocab.py --pdfs /path/pdf --limit 800 \
        --out ../reading_build/vocab.txt
    python3 build_vocab.py --jsonl /root/sft2.jsonl \
        --out ../reading_build/vocab.txt

The jsonl form reads the SFT pages instead of PDFs: the corpus PDFs are
deleted once harvested, and every line of every page is already in the file.
"""
import argparse
import json
import os
import re
import sys

TOKEN = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


def count_text(t, cnt):
    # a trailing "-" is a break, not a word
    for w in TOKEN.findall(t):
        w = w.strip("'’-").lower()
        if len(w) >= 2:
            cnt[w] = cnt.get(w, 0) + 1


def count_jsonl(path, cnt, min_count):
    """Count the line texts of chat rows, not the raw JSON, so that schema
    keys ("bbox", "role") never enter the vocabulary."""
    rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            for m in json.loads(line).get("messages") or []:
                if m.get("role") != "user":
                    continue
                try:
                    src = json.loads(m.get("content") or "{}")
                except json.JSONDecodeError:
                    continue
                for ln in src.get("lines") or []:
                    if len(ln) >= 8 and isinstance(ln[7], str):
                        count_text(ln[7], cnt)
            if rows % 2000 == 0:
                print(f"  {rows} rows, {len(cnt)} tokens", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs")
    ap.add_argument("--jsonl")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (a.pdfs or a.jsonl):
        ap.error("--pdfs DIR or --jsonl FILE is required")

    cnt = {}
    if a.jsonl:
        rows = count_jsonl(a.jsonl, cnt, a.min_count)
        keep = {w: c for w, c in cnt.items()
                if c >= a.min_count and w.isascii()}
        with open(a.out, "w", encoding="utf-8") as f:
            for w, c in sorted(keep.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"{w} {c}\n")
        print(f"{rows} rows, {len(cnt)} tokens, {len(keep)} kept -> {a.out}",
              flush=True)
        return 0

    files = sorted(f for f in os.listdir(a.pdfs) if f.endswith(".pdf"))
    step = max(1, len(files) // a.limit)
    files = files[::step][:a.limit]
    for i, name in enumerate(files, 1):
        try:
            doc = pymupdf.open(os.path.join(a.pdfs, name))
        except Exception as e:                            # noqa: BLE001
            print(f"[{i}] {name} {type(e).__name__}: {e}"[:120], flush=True)
            continue
        for page in doc:
            for b in page.get_text("dict")["blocks"]:
                for l in b.get("lines", []):
                    t = "".join(s["text"] for s in l["spans"])
                    count_text(t, cnt)
        doc.close()
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    keep = {w: c for w, c in cnt.items() if c >= a.min_count}
    with open(a.out, "w", encoding="utf-8") as f:
        for w, c in sorted(keep.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{w} {c}\n")
    print(f"{len(files)} papers, {len(cnt)} tokens, {len(keep)} kept "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
