#!/usr/bin/env python3
"""Serialise the rule converter's decisions into (input lines, labels) pairs.

Input is intentionally geometric.  The converter reads the page in its own
order and the PDF content stream carries that order; feeding it to the model
would hand over the answer.  Lines are therefore sorted by a naive
top-to-bottom, left-to-right sweep, which interleaves the columns of a
two-column paper and leaves the hard part to the model.

Label granularity is the typeset line.  The converter works on blocks that
are paragraphs on one page and single lines on the next, so its blocks are
split back into the lines they are made of; `Item.group` says which blocks
each emitted paragraph absorbed, which gives every line its paragraph id and
role exactly rather than by matching text.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

# the sibling checkout of the renderer, wherever this tree lives
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "reading_build"))
import build  # noqa: E402


def record_graphics(pd):
    """The page's artwork as boxes, including rule-built tables.

    `grects` deliberately skips zero-thickness drawings; `lrects` is their
    clustered form (a table drawn with `\\hline` has no other artwork).  A
    caption without artwork renders as text alone, so the table would vanish
    from the page.
    """
    rects = [tuple(r) for r in pd["grects"]] + [tuple(r) for r in pd["imgs"]]
    rects += [(r.x0, r.y0, r.x1, r.y1) for r in pd.get("lrects", ())]
    return [[round(v, 1) for v in r] for r in rects]


def emit(pdf, outroot, dpi):
    res = build.convert(pdf, Path(outroot), dpi=dpi)
    pages = res["pages"]
    merged = res["items"]

    role = {}                       # id(item) -> (role, paragraph index)
    readpos = {}                    # id(item) -> reading position of line 0
    paras = []
    n = 0
    for pi, (_pd, it) in enumerate(merged):
        paras.append({"role": it.kind, "n": len(it.group),
                      "text": " ".join(it.text.split())[:120]})
        for sub in it.group:
            role[id(sub)] = (it.kind, pi)
            # The group is in reading order: a paragraph continued at the
            # top of the next column has its head appended first.  The
            # sweep below destroys that order, and a label that joins a
            # block in sweep order inside a column break reads backwards.
            #
            # Positions count *lines*, not items: the record is written one
            # entry per typeset line (see below), so a paragraph keeps a
            # contiguous range and its lines stay ordered inside it.
            readpos[id(sub)] = n
            n += max(1, len(sub.lines))

    # Front matter is metadata, not structure: the converter already finds it,
    # the model is not asked for it, and the renderer needs a title that is
    # not the largest heading that happened to survive on page 0.  Only the
    # strings are kept — the converter's record carries object ids.
    fm = res["fm"] or {}
    out = {"doc": os.path.basename(pdf)[:-4],
           "fm": {"title": fm.get("title", ""),
                  "authors": fm.get("authors", "")},
           "npages": len(pages), "paras": paras, "pages": []}
    for pno, pd in enumerate(pages):
        items = []
        for it in pd["items"]:
            r, pi = role.get(id(it), ("drop", -1))
            base = readpos.get(id(it), -1)
            # One entry per typeset line, not per converter block: the
            # converter's blocks are paragraphs on some pages and single
            # lines on others (it depends on how the PDF camera splits the
            # page), and a training set that changes its unit from page to
            # page teaches the model that the answer depends on something
            # it cannot see.  Lines are the unit the converter itself
            # merges, so the labels are exactly its paragraph decisions.
            # The converter locates each caption's artwork as part of the
            # same pass that finds the caption.  Dropping that box and
            # re-deriving it downstream is how a page rule ends up inside the
            # crop: the renderer only has rectangles, while the converter has
            # the item list, the column layout and the text it must not cross.
            reg = getattr(it, "region", None)
            fbox = None if reg is None else [round(reg.x0, 1), round(reg.y0, 1),
                                             round(reg.x1, 1), round(reg.y1, 1)]
            for li, (txt, sz, bold, bb) in enumerate(it.lines):
                t = " ".join(txt.replace("\u00ad", "-").split())
                if not t:
                    continue
                rec = {
                    "bbox": [round(v, 1) for v in bb],
                    "text": t,
                    "size": round(sz, 1),
                    "bold": bool(bold),
                    "role": r, "para": pi, "lvl": it.level,
                    "ord": -1 if base < 0 else base + li,
                }
                if fbox and li == 0:
                    rec["fig"] = fbox
                items.append(rec)
        # geometric sweep, deliberately not reading order — but each line
        # still carries its reading position, which is what orders the
        # members of a block that the sweep splits apart
        order = sorted(range(len(items)),
                       key=lambda k: (round(items[k]["bbox"][1] / 2), k))
        out["pages"].append({
            "page": pno,
            "page_size": [round(pd["size"][0], 1),
                          round(pd["size"][1], 1)],
            "graphics": record_graphics(pd),
            "lines": [items[k] for k in order],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True)
    # repeatable and globbable: QC runs in shards, each writing its own file
    ap.add_argument("--qc", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--dpi", type=int, default=24)
    # `--shard K/N` takes every N-th passing paper; shards write into the
    # same --out safely because a record's name is its paper's name
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()

    keep = set()
    for pat in a.qc:
        for path in glob.glob(pat):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r.get("pass"):
                    keep.add(r["pdf"])
    keep = sorted(keep)
    sk, sn = (int(x) for x in a.shard.split("/"))
    keep = keep[sk::sn]
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(keep)} passing papers", flush=True)

    n = 0
    for i, name in enumerate(keep, 1):
        dst = os.path.join(a.out, name[:-4] + ".json")
        if os.path.exists(dst):
            n += 1
            continue
        try:
            rec = emit(os.path.join(a.pdfs, name), a.tmp, a.dpi)
        except Exception as e:                            # noqa: BLE001
            print(f"[{i}] {name} ERR {type(e).__name__}: {e}"[:160], flush=True)
            continue
        # Two emit passes over the same QC set are harmless — each record is
        # named after its paper and an existing one is skipped — but only if a
        # half-written file can never be seen: the rebuild driver and a
        # pre-fill pass can name the same paper, and the loser must not leave
        # truncated JSON behind for to_sft to trip on.
        # per-process suffix: two passes may name the same record at the same
        # moment, and a shared temp name makes the loser's rename fail after
        # the winner already moved the file out from under it
        tmp = f"{dst}.part.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        os.replace(tmp, dst)
        n += 1
        if i % 50 == 0:
            print(f"  {i}/{len(keep)}", flush=True)
    print(f"wrote {n} records to {a.out}", flush=True)


if __name__ == "__main__":
    main()
