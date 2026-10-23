#!/usr/bin/env python3
"""Structure JSON -> reading-edition HTML.

The model (or the gold teacher) says what each line is and in what order;
this file turns that into a page.  Every visual decision lives in TASTE.md and
theme.css, so the renderer is deliberately boring: it maps roles to
components and nothing else.

    python3 render.py --pdf X.pdf --rec out/X.json --labels labels.json --out outdir

`--rec` is the page.json that emit.py writes (the same file the model was fed),
`--labels` is the assistant array, `--pdf` is only used to crop figures.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

esc = build.esc
join_lines = build.join_lines
from _regions import columns  # noqa: E402

def split_head(txt):
    """'III. METHOD' -> ('III', 'METHOD'); '2.1 Setup' -> ('2.1', 'Setup')."""
    for pat in (build.HEAD_PAT, build.ROMAN_PAT, build.LETTER_PAT):
        m = pat.match(txt.strip())
        if m:
            return m.group(1).rstrip("."), m.group(2)
    return "", txt
# The converter's own caption parser is the authority: it accepts "TABLE I:"
# (roman numerals, upper case) and rejects "Figure 10 shows that ...".
CAP = build.parse_caption


def slug(s, n=24):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s[:n] or "sec")


def short_title(t, limit=110):
    """The masthead title, cut at a clause boundary when it is very long.

    A six-line masthead pushes the article below the fold; a title cut
    mid-word reads as a bug.  Cut at the last colon, comma or dash before
    the limit, and keep the full string in `<title>` and in the `title`
    attribute.
    """
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= limit:
        return t
    best = 0
    for m in re.finditer(r"[:;,—–] |, ", t[:limit]):
        best = m.end() - 1
    cut = t[:best] if best >= 40 else t[:limit]
    return cut.rsplit(" ", 1)[0].rstrip(",;:.") + "…"


def gold_labels(pg):
    """Reconstruct the teacher's grouping from the record (same rule as to_sft)."""
    order = {}
    for pos, ln in enumerate(pg["lines"]):
        if ln["para"] >= 0:
            order.setdefault(ln["para"], []).append(pos)
    out = []
    for pi in sorted(order):
        # reading order, not the record's top-to-bottom sweep: a block that
        # crosses the column gutter would otherwise be joined tail-first
        mem = sorted(order[pi], key=lambda i: pg["lines"][i].get("ord", i))
        ln = pg["lines"][mem[0]]
        r = ln["role"]
        if r == "h":
            r = "h1" if ln.get("lvl", 1) <= 1 else "h2"
        out.append({"r": r if r in ("h1", "h2", "p", "cap", "kw", "small")
                    else "drop", "l": mem})
    placed = {i for b in out for i in b["l"]}
    rest = [i for i in range(len(pg["lines"])) if i not in placed]
    out.extend({"r": "drop", "l": [i]} for i in rest)
    return out


def blocks_of(rec, labels):
    """Flatten labels into (page, role, text, first-line bbox)."""
    out = []
    for pno, pg in enumerate(rec["pages"]):
        lines = pg["lines"]
        lab = labels[pno] if pno < len(labels) else []
        for b in lab:
            idx = [i for i in b.get("l", []) if 0 <= i < len(lines)]
            if not idx:
                continue
            txt = join_lines([lines[i]["text"] for i in idx])
            if not txt:
                continue
            out.append({"page": pno, "r": b.get("r", "p"), "text": txt,
                        "idx": idx, "box": lines[idx[0]]["bbox"],
                        "size": lines[idx[0]]["size"]})
    return out


def graphics_near(rec, pno, box, kind):
    """Rects on this page that belong with a caption at `box`.

    "Near" is not enough.  A rule or a colour block from the other column, or
    the previous figure on this one, can sit within the same y window; a
    union over everything near turns the crop into a strip of the whole page.
    So a candidate must share at least half its width with the caption, and
    for a figure the crop grows upward from the caption only while the rects
    stay in one cluster — the first gap ends the figure.
    """
    x0, y0, x1, y1 = box
    cw = max(x1 - x0, 1.0)
    cand = []
    for g in rec["pages"][pno]["graphics"]:
        gx0, gy0, gx1, gy1 = g
        gw = max(gx1 - gx0, 1.0)
        ov = max(0.0, min(x1, gx1) - max(x0, gx0))
        if ov / min(cw, gw) < 0.5:
            continue
        # A page rule under the title runs the full text width and can sit
        # within the caption's y window; unioned in, it drags the crop across
        # the gutter and the "figure" comes out with the other column's prose
        # printed beside it.  Extent alone cannot separate it from a table
        # rule, which is equally thin and equally wide while the caption
        # above it is three words long ("Table 1").  Distance can: a table's
        # rules start a few points below its caption, a page rule is far away.
        if gy1 - gy0 < 3.0 and min(abs(gy1 - y0), abs(y1 - gy0)) > 60:
            continue
        cand.append(g)
    if kind == "Table":
        # a table caption sits above its rules
        return [g for g in cand if y1 - 8 <= g[1] <= y1 + 90]
    near, top = [], y0 + 8
    for g in sorted(cand, key=lambda g: -g[3]):
        if top - g[3] > 45:
            break                       # a gap: another figure, not this one
        near.append(g)
        top = min(top, g[1])
    return near


def caption_seqno(page, box):
    """Content-stream position of the text on this line, or None.

    PDF drawings carry a sequence number in the order they were painted, and
    a caption is painted after the artwork it labels.  That ordering is the
    one piece of structure geometry cannot see: on a dense page every vector
    mark is within a few points of some other vector mark, so geometric
    clustering merges the whole page into one blob and the crop swallows the
    body text around it.
    """
    best = None
    try:
        spans = page.get_texttrace()
    except Exception:                                        # noqa: BLE001
        return None
    for t in spans:
        b = t.get("bbox") or ()
        if len(b) < 4:
            continue
        if abs(b[1] - box[1]) < 3 and abs(b[0] - box[0]) < 40:
            s = t.get("seqno")
            if s is not None:
                best = s if best is None else min(best, s)
    return best


def seq_rect(page, rec_page, box, cols):
    """Artwork belonging to a caption, read off the paint order."""
    sq = caption_seqno(page, box)
    if sq is None:
        return None
    x0, y0, x1, y1 = box
    home = None
    if cols:
        home = min(cols, key=lambda c: abs((c[0] + c[1]) / 2.0 - (x0 + x1) / 2.0))
    sel = []
    try:
        draws = page.get_drawings()
    except Exception:                                        # noqa: BLE001
        return None
    for dr in draws:
        r = dr.get("rect")
        if r is None or dr.get("seqno", 0) >= sq:
            continue
        if home and (r.x1 < home[0] - 12 or r.x0 > home[1] + 12):
            continue
        if r.y1 > y0 + 6:
            continue
        sel.append(r)
    if not sel:
        return None
    sel.sort(key=lambda r: -r.y1)
    out, edge = [sel[0]], sel[0].y0
    for r in sel[1:]:
        if edge - r.y1 > 60:
            break
        out.append(r)
        edge = min(edge, r.y0)
    return (min(r.x0 for r in out), min(r.y0 for r in out),
            max(r.x1 for r in out), max(r.y1 for r in out))


def union(rects):
    if not rects:
        return None
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def crop(page, rect, dest, dpi=200):
    if rect is None:
        return None
    pad = 2
    r = pymupdf.Rect(rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad)
    r = pymupdf.Rect(max(r.x0, 0), max(r.y0, 0),
                     min(r.x1, page.rect.width), min(r.y1, page.rect.height))
    if r.height < 14 or r.width < 24:
        return None
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                          clip=r, alpha=False)
    if pix.width > 1600:
        s = 1600 / pix.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi * s / 72, dpi * s / 72),
                              clip=r, alpha=False)
    pix.save(str(dest))
    return pix.width, pix.height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--authors", default="")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rec = json.load(open(a.rec, encoding="utf-8"))
    if a.labels == "gold":                 # self-test: the teacher's own labels
        labels = [gold_labels(pg) for pg in rec["pages"]]
    else:
        labels = json.load(open(a.labels, encoding="utf-8"))

    doc = pymupdf.open(a.pdf)
    bl = blocks_of(rec, labels)

    # The converter's front matter is the title; the largest kept block on
    # page 0 is usually the first section heading, and a paper titled
    # "I. INTRODUCTION" is worse than no title at all.  The block heuristic
    # stays only as a last resort for records made without front matter.
    fm = rec.get("fm") or {}
    title = a.title or fm.get("title", "")
    if not title:
        best = max((b for b in bl if b["page"] == 0), key=lambda b: b["size"],
                   default=None)
        title = best["text"] if best else Path(a.pdf).stem
    authors = a.authors or fm.get("authors", "")

    toc, flow = [], []
    fig_n = 0
    open_sec = None
    after_abstract = False
    first_para_done = False
    # A paragraph that a page break cut in half is one paragraph; emitting it
    # as two leaves the reader with a sentence that stops mid-word and a new
    # one that starts mid-word.  Remember the last paragraph while it is still
    # the tail of its page, and stitch the next page's opening onto it.
    tail_p = None            # (flow index, page, text) or None

    for b in bl:
        role, txt = b["r"], b["text"]

        if role in ("drop", "title"):
            continue

        if role not in ("small", "drop", "p"):
            tail_p = None
        if role in ("h1", "h2"):
            num, name = split_head(txt)
            name = name.rstrip(" .:")
            if not name or len(name) > 90:
                continue
            sid = f"s{len(toc) + 1}-{slug(name)}"
            lvl = 1 if role == "h1" else 2
            toc.append((lvl, sid, name, num))
            tag = "h2" if lvl == 1 else "h3"
            cls = "sec" if lvl == 1 else "sub"
            nspan = f'<span class="n">{esc(num)}</span> ' if num else ""
            flow.append(f'<{tag} class="{cls}" id="{sid}">{nspan}'
                        f'{esc(name)}</{tag}>')
            open_sec = name
            if name.lower() == "abstract":
                after_abstract = True
            continue

        if role == "cap":
            m = CAP(txt)
            if not m:
                flow.append(f'<p class="rawcap">{esc(txt)}</p>')
                continue
            kind, num, rest = m
            kind = "Table" if kind.startswith("t") else "Figure"
            fig_n += 1
            # The record carries the converter's own box.  It is the best
            # signal available and it costs nothing: it was computed when the
            # caption was found.  The rest of the chain exists for records
            # made before that box was kept, and for captions the converter
            # recognised without locating artwork for.
            rect = rec["pages"][b["page"]]["lines"][b["idx"][0]].get("fig")
            if rect is None:
                rect = seq_rect(doc[b["page"]], rec["pages"][b["page"]],
                                b["box"], columns(rec["pages"][b["page"]]))
            if rect is None:
                rects = graphics_near(rec, b["page"], b["box"], kind)
                rect = union(rects) if rects else None
            dest = out / "img" / f"g{fig_n:02d}.png"
            size = crop(doc[b["page"]], rect, dest, a.dpi) if rect else None
            capnum = f'<span class="capnum">{esc(kind)} {esc(num)}.</span>'
            if size:
                w, h = size
                alt = esc(re.sub(r"<[^>]+>", "", rest)[:180]) or kind
                # the class names are theme.css's contract: a figure is a
                # white-background raster that gets inverted inside a dark
                # plate, a table is a raster that scrolls on its own axis
                cls = "tablewrap" if kind == "Table" else "plate__frame"
                img = (f'<div class="{cls}">'
                       f'<img src="img/{dest.name}" alt="{alt}" '
                       f'width="{w}" height="{h}" loading="lazy"></div>')
            else:
                img = ""
            fcls = "table" if kind == "Table" else "plate"
            flow.append(f'<figure class="{fcls}">{img}<figcaption>{capnum} '
                        f'{esc(rest)}</figcaption></figure>')
            continue

        if role == "kw":
            flow.append(f'<p class="kw">{esc(txt)}</p>')
            continue

        if role == "small":
            # A running head sits in the top margin and repeats the title or
            # the author line on every page.  Rendered in flow it lands in the
            # middle of a paragraph the page break split, which reads as a
            # stray line of the article.  Footnotes live at the bottom and are
            # unaffected.
            ps = rec["pages"][b["page"]].get("page_size") or (0, 0)
            ph = ps[1] if len(ps) > 1 else 0
            if ph and b["box"][3] < 0.08 * ph:
                continue
            flow.append(f'<p class="fnl">{esc(txt)}</p>')
            continue

        if tail_p is not None and tail_p[1] != b["page"] and \
                not tail_p[2].rstrip().endswith((".", "!", "?", ":", ";")) and \
                txt[:1].islower():
            i = tail_p[0]
            flow[i] = flow[i][:-4] + " " + esc(txt) + "</p>"
            tail_p = (i, b["page"], tail_p[2] + " " + txt)
            continue
        attrs = ""
        if after_abstract and not first_para_done:
            # `.opening` is theme.css's drop cap and only theme.css's
            attrs, first_para_done = ' class="opening"', True
        flow.append(f"<p{attrs}>{esc(txt)}</p>")
        tail_p = (len(flow) - 1, b["page"], txt)

    # the tab title carries the paper, not the subtitle after a colon
    short = re.sub(r":.*$", "", title).strip() or title
    if len(short) > 72:
        short = short[:69].rstrip() + "…"
    mast = short_title(title)
    css = (HERE / "theme.css").read_text(encoding="utf-8")

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(short)} · reading edition</title>",
        "<style>\n" + css + "</style>",
        "</head>",
        "<body>",
        '<div class="toprule"></div>',
        '<div class="shell">',
        '  <aside class="side">',
        '    <div class="toc">',
        '      <p class="toc__label">Contents</p>',
        '      <button class="toc-toggle" type="button" aria-expanded="false" '
        'aria-controls="toc">Contents ▾</button>',
        '      <nav id="toc"><ol>',
    ]
    for lvl, sid, name, num in toc[:120]:
        label = (num + " " + name).strip()
        parts.append(f'        <li><a href="#{sid}"'
                     + (' class="toc__sub"' if lvl == 2 else "")
                     + f'>{esc(label[:110])}</a></li>')
    parts += [
        "      </ol></nav>",
        f'      <p class="toc__aside"><strong>Source</strong>'
        f'{esc(Path(a.pdf).name)}</p>',
        "    </div>",
        "  </aside>",
        '  <main class="paper">',
        '    <header class="mast">',
        f'      <p class="mast__kicker">Re-typeset from PDF · '
        f'{len(rec["pages"])} pages</p>',
        f'      <h1 class="title" title="{esc(title)}">{esc(mast)}</h1>',
    ]
    if authors:
        parts.append(f'      <p class="authors">{esc(authors)}</p>')
    parts += [
        "    </header>",
        '    <div class="flow">',
        "\n".join("      " + x for x in flow),
        "    </div>",
        "  </main>",
        "</div>",
        "</body>",
        "</html>",
    ]
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out / 'index.html'} ({len(flow)} blocks, {fig_n} figures)")


if __name__ == "__main__":
    main()
