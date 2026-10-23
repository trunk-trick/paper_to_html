#!/usr/bin/env python3
"""Turn emitted records into chat-format SFT examples.

One example is one page.  Lines go in as fixed-position arrays (index, bbox,
size, bold, text) because objects cost roughly twice the tokens for the same
information; the schema is stated once in the system prompt.

The target is a reading-ordered list of groups.  Order is the point: the input
is a top-to-bottom, left-to-right sweep, so a two-column page arrives
interleaved and the model has to undo that.
"""
import argparse
import json
import os
import random

SYSTEM = (
    "You reconstruct the reading structure of one page of an academic paper. "
    "The input is a JSON object with the page size, the rectangles of figures "
    "and tables on the page (graphics), and the text lines in a naive "
    "top-to-bottom, left-to-right sweep that is NOT reading order. Each line "
    "is [index, x0, y0, x1, y1, font size, bold, text].\n"
    "Return a JSON array of the page's blocks in true reading order. Each "
    "block is {\"r\": role, \"l\": [line indices]}. Roles:\n"
    "h1 major section heading, h2 subsection heading, p body paragraph, "
    "cap figure or table caption, kw keywords/CCS block, small footnote or "
    "reference entry, drop anything not part of the article (running head, "
    "page number, publisher boilerplate, text inside a figure or table).\n"
    "Every line index must appear exactly once. Output the array only."
)

SHORT = {"p": "p", "h": "h", "cap": "cap", "kw": "kw", "small": "small",
         "drop": "drop"}


def role_of(line):
    r = line["role"]
    if r == "h":
        return "h1" if line.get("lvl", 1) <= 1 else "h2"
    return SHORT.get(r, "drop")


def example(rec, pno):
    pg = rec["pages"][pno]
    lines, groups = [], []
    order = {}
    for pos, ln in enumerate(pg["lines"]):
        x0, y0, x1, y1 = ln["bbox"]
        lines.append([pos, round(x0), round(y0), round(x1), round(y1),
                      ln["size"], 1 if ln["bold"] else 0, ln["text"]])
        if ln["para"] >= 0:
            order.setdefault(ln["para"], []).append(pos)

    for pi in sorted(order):
        # a block crossing the column gutter is not contiguous in the
        # top-to-bottom sweep; its members go out in reading order
        members = sorted(order[pi], key=lambda i: pg["lines"][i].get("ord", i))
        first = pg["lines"][members[0]]
        groups.append({"r": role_of(first), "l": members})

    # Lines the converter never placed in a paragraph are dropped.  They go
    # out as one block, not one block per line: every line must appear
    # exactly once, so a one-index block per dropped line spends most of the
    # answer on the part the renderer throws away, and on a figure-heavy
    # page that is half the array.
    placed = {p for g in groups for p in g["l"]}
    rest = [i for i in range(len(pg["lines"])) if i not in placed]
    if rest:
        groups.append({"r": "drop", "l": rest})

    src = {"page": pg["page"], "size": pg["page_size"],
           "graphics": pg["graphics"], "lines": lines}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(src, ensure_ascii=False)},
            {"role": "assistant",
             "content": json.dumps(groups, ensure_ascii=False)},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    # A paper with an appendix can be 60 pages; uncapped, a handful of long
    # papers becomes a genre of its own in the training mix.  Pages are
    # independent examples, so an even sample costs nothing.
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--source", default="")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    files = sorted(f for f in os.listdir(a.src) if f.endswith(".json"))
    nval = max(1, int(len(files) * a.val_frac))
    rows = []
    for name in files:
        rec = json.load(open(os.path.join(a.src, name), encoding="utf-8"))
        pnos = list(range(len(rec["pages"])))
        if a.max_pages and len(pnos) > a.max_pages:
            pnos = sorted(rng.sample(pnos, a.max_pages))
        for pno in pnos:
            rows.append((name, example(rec, pno)))

    # split by paper, never by page: pages of one paper are near-duplicates.
    # The sample is drawn by seed, not by file-name prefix: paper names sort
    # into blocks (a generator's classes, an arXiv run's month), and taking
    # the first 5% would hold out one class as the whole validation set.
    val_names = set(rng.sample(files, nval))
    src = a.source or os.path.basename(os.path.abspath(a.src))
    with open(a.out, "w", encoding="utf-8") as f:
        for name, ex in rows:
            ex["split"] = "val" if name in val_names else "train"
            ex["doc"] = name
            ex["source"] = src
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{len(rows)} pages from {len(files)} papers "
          f"({nval} papers held out) -> {a.out}")


if __name__ == "__main__":
    main()
