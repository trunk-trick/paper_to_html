"""Candidate artwork regions on a page, and the caption that owns them.

Artwork arrives from the converter as a flat list of rectangles: four panels
of one figure, each of their internal strokes, the rules of a table, and the
hairline rules that separate the title block from the body.  The renderer
needs "the picture this caption names", which is a decision about groups of
rectangles, not about single ones -- unioning raw rectangles lets one stray
page rule stretch the crop to the full text width.

  regions(pg)         rectangles -> connected clusters, each with an id
  assign(pg, box, k)  caption + kind -> the cluster ids that make up its
                      artwork, chosen by column, distance and direction

The ids are what a model can be asked to predict; the boxes are what the
crop needs.
"""
GAP = 8.0            # rects closer than this belong to one piece of artwork
CHAIN = 60.0         # how far the artwork may sit from its caption


def regions(pg):
    """Connected components of the page's artwork, top down then left right."""
    rects = [tuple(float(v) for v in g) for g in pg.get("graphics", [])]
    n = len(rects)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        ax0, ay0, ax1, ay1 = rects[i]
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = rects[j]
            if not (ax1 + GAP < bx0 or bx1 + GAP < ax0 or
                    ay1 + GAP < by0 or by1 + GAP < ay0):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(rects[i])
    out = []
    for mem in groups.values():
        out.append({"bbox": [min(r[0] for r in mem), min(r[1] for r in mem),
                             max(r[2] for r in mem), max(r[3] for r in mem)],
                    "n": len(mem)})
    out.sort(key=lambda r: (round(r["bbox"][1] / 10), r["bbox"][0]))
    for i, r in enumerate(out):
        r["id"] = i
    return out


def columns(pg):
    """Text columns on the page, as (x0, x1) pairs, left to right.

    A caption on a two-column page belongs to the column it was set in, and
    the artwork it names is in that column too.  Without this a rule or a
    figure that spans the gutter joins the cluster, the crop grows across the
    page, and the reader gets the other column's prose printed beside the
    picture.
    """
    lines = [l["bbox"] for l in pg.get("lines", [])
             if l.get("text") and l["role"] not in ("drop",)]
    if not lines:
        return {"ids": [], "box": None}
    xs = [b[0] for b in lines]
    lo, hi = min(xs), max(xs)
    if hi - lo < 80:
        return [(lo, max(b[2] for b in lines))]
    mid = (lo + hi) / 2.0
    left = [b for b in lines if b[0] < mid]
    right = [b for b in lines if b[0] >= mid]
    # both sides need enough lines to be a column, not a stray indent
    if len(left) < 0.25 * len(lines) or len(right) < 0.25 * len(lines):
        return [(lo, max(b[2] for b in lines))]
    # a real gutter: the right column starts well past where the left ends
    lx1 = sorted(b[2] for b in left)[int(0.9 * (len(left) - 1))]
    rx0 = sorted(b[0] for b in right)[int(0.1 * (len(right) - 1))]
    if rx0 - lx1 < 10:
        return [(lo, max(b[2] for b in lines))]
    return [(lo, lx1), (rx0, max(b[2] for b in right))]


def _ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _vgap(a, b):
    """Vertical clearance between two boxes; 0 when their spans overlap."""
    return max(0.0, max(a[1], b[1]) - min(a[3], b[3]))


def assign(pg, box, kind, regs=None):
    """Cluster ids making up the artwork this caption describes.

    Direction is not fixed by kind: journals put figure captions under the
    artwork and table captions over it, but camera-ready PDFs break both
    conventions often enough that assuming one empties a third of the crops.
    So take the nearest cluster as the anchor on whichever side it lies, and
    chain outward from it while the artwork stays contiguous.
    """
    regs = region_list(pg, regs)
    if not regs:
        return {"ids": [], "box": None}
    x0, y0, x1, y1 = box
    cw = max(x1 - x0, 1.0)
    all_regs = regs
    # keep only artwork that lies in the caption's own column
    cols = columns(pg)
    if cols:
        home = min(cols, key=lambda c: abs((c[0] + c[1]) / 2.0 - (x0 + x1) / 2.0))
        pad = 12.0
        regs = [r for r in regs
                if _ov(home[0] - pad, home[1] + pad,
                       r["bbox"][0], r["bbox"][2]) >=
                0.6 * max(r["bbox"][2] - r["bbox"][0], 1.0)]
    def in_col(rs):
        return [r for r in rs
                if _ov(x0, x1, r["bbox"][0], r["bbox"][2]) >=
                0.25 * min(cw, max(r["bbox"][2] - r["bbox"][0], 1.0))]

    col = in_col(regs)
    if not col:
        # a full-width figure whose caption sits in one column: the column
        # filter has thrown away the only candidates, so drop it rather than
        # render the figure as nothing
        col = in_col(all_regs)
    if not col:
        return {"ids": [], "box": None}

    def gap(r):                     # vertical distance from the caption box
        return _vgap(r["bbox"], box)

    # Anchor on the nearest cluster, on whichever side it lies.  Classifying
    # a cluster as "above" by whether its bottom beats the caption's top fails
    # on the ordinary case where the artwork's bounding box overlaps the
    # caption line by a point or two -- the anchor then falls out of both
    # lists and the figure renders empty.  Compare centres instead.
    above = [r for r in col
             if (r["bbox"][1] + r["bbox"][3]) < (y0 + y1)]
    below = [r for r in col
             if (r["bbox"][1] + r["bbox"][3]) >= (y0 + y1)]
    first = above if kind == "Figure" else below
    anchor = min(first or below or above, key=gap)
    up = (anchor["bbox"][1] + anchor["bbox"][3]) < (y0 + y1)
    side = above if up else below
    side.sort(key=lambda r: -r["bbox"][3] if up else r["bbox"][1])
    out, edge = [], None
    for r in side:
        g = gap(r) if edge is None else (
            max(0.0, edge - r["bbox"][3]) if up
            else max(0.0, r["bbox"][1] - edge))
        if out and g > CHAIN:
            break
        out.append(r)
        edge = r["bbox"][1] if up else r["bbox"][3]
    # The chosen artwork can still straddle the gutter -- a rule that runs the
    # full text width gets drawn as part of the figure.  Crop it back to the
    # caption's column instead of either discarding the figure or printing the
    # neighbouring column beside it.
    box_out = [min(r["bbox"][0] for r in out), min(r["bbox"][1] for r in out),
               max(r["bbox"][2] for r in out), max(r["bbox"][3] for r in out)]
    if cols:
        home = min(cols, key=lambda c: abs((c[0] + c[1]) / 2.0 - (x0 + x1) / 2.0))
        colw = home[1] - home[0]
        capw = x1 - x0
        if box_out[2] - box_out[0] > 1.15 * max(colw, 1.0) and capw < 0.9 * colw:
            box_out[0] = max(box_out[0], home[0] - 12.0)
            box_out[2] = min(box_out[2], home[1] + 12.0)
    return {"ids": [r["id"] for r in out], "box": box_out}


def region_list(pg, regs=None):
    return regs if regs is not None else regions(pg)
