#!/usr/bin/env python3
"""Batch converter: academic PDF -> dark HTML reading edition.

Usage:  python3 -P build.py [--only SUBSTR] [--dpi 200] [--out DIR]

Design constraints, learned the hard way on the hand-built pages:
  * the CSS comes verbatim from FDP_paper.html (same series, same tokens);
  * figures are cropped from the page rather than extracted as embedded
    images, because in these papers the axis labels and legends are vector
    text outside the raster/plot area;
  * a figure has no title of its own, so the crop is bounded by the nearest
    text above it and by its caption below -- never by fixed padding.
"""
import argparse, html, os, re, sys, unicodedata
from pathlib import Path

import pymupdf

FDP = Path(os.environ.get("P2H_FDP") or
           Path(__file__).with_name("FDP_paper.html"))

# ---------------------------------------------------------------- utilities

def clean(s):
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.replace("­", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def esc(s):
    return html.escape(s, quote=False)


_VOCAB = None


def vocab():
    """Word -> count, built from the corpus by corpus/build_vocab.py."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = {}
        try:
            txt = Path(__file__).with_name("vocab.txt").read_text(encoding="utf-8")
            for line in txt.splitlines():
                w, _, c = line.rpartition(" ")
                if w:
                    _VOCAB[w] = int(c)
        except (OSError, ValueError):
            pass                    # no vocabulary: fall back to plain joining
    return _VOCAB


_HEAD = re.compile(r"([A-Za-z][A-Za-z'’-]*)-$")
_TAIL = re.compile(r"([A-Za-z][A-Za-z'’-]*)")


def compound(head, tail):
    """Whether `head-tail` is one word that merely broke at its hyphen.

    "signal-" + "flow" is a compound: both halves are common words and the
    run-together form is not.  "guaran-" + "tee" is a typesetter's break: the
    first half is not a word at all.  Around eighty line ends per paper are
    one or the other, so guessing wrong is visible on every page.
    """
    v = vocab()
    if not v:
        return False
    # "state-of-the-" breaks at "the", not at "state": the segment that has
    # to be a word is the one the tail would attach to
    h = head.rsplit("-", 1)[-1].lower()
    t = tail.lower()
    if len(h) < 2 or len(t) < 2:
        return False
    # run together, it is a word ("com"+"pared" -> "compared"): the hyphen
    # was a line break, whatever the halves look like on their own
    if v.get(h + t, 0) >= 5:
        return False
    # otherwise it is a compound only if both halves are words.  The sample
    # is 800 papers, so a real domain term is rare here ("trojan" 5) while
    # an actual fragment is commoner than it should be ("cessing" 74, from
    # the beginnings of broken lines); "joined is a word" carries the rest.
    return v.get(h, 0) >= 3 and v.get(t, 0) >= 3


def join_lines(lines):
    """Join the lines of one paragraph, undoing end-of-line hyphenation."""
    out = ""
    for t in lines:
        # PDFs mark a typeset break with U+00AD, not "-"; left alone it is an
        # invisible character glued to the end of the word ("un\xad" + "derwear"
        # -> "un derwear"), which reads as a typo on every hyphenated line.
        t = t.strip().replace("\u00ad", "-")
        if not t:
            continue
        if not out:
            out = t
            continue
        if out.endswith("-") and len(out) > 1 and t[:1].islower():
            mh, mt = _HEAD.search(out), _TAIL.match(t)
            if mh and mt and compound(mh.group(1), mt.group(1)):
                out = out + t          # the hyphen belongs to the word
            else:
                out = out[:-1] + t
        elif out.endswith("-") and len(out) > 1 and not t[:1].isalnum():
            out = out[:-1] + t
        else:
            out = out + " " + t
    return clean(out)


def is_bold(font, flags):
    f = font.lower()
    return ("bold" in f or "black" in f or "heavy" in f or "medi" in f
            or bool(flags & 16))


HEAD_PAT = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(\S.*)$")
ROMAN_PAT = re.compile(r"^([IVX]{1,4})\.\s+(\S.*)$")
LETTER_PAT = re.compile(r"^([A-Z])\.\s+(\S.*)$")
CAP_PAT = re.compile(r"^(Figure|Fig\.?|FIGURE|Table|TABLE)\s*([0-9]{1,3}|[IVX]{1,4})"
                     r"\s*([:.\u2013\u2014-]?)\s*(.*)$")
# "Figure 10 shows that ..." is a sentence about a figure, not a caption
CAP_VERB = re.compile(r"^(shows|show|illustrates?|depicts?|presents?|compares?|"
                      r"reports?|plots?|gives?|lists?|is|are|and)\b", re.I)


def parse_caption(t):
    m = CAP_PAT.match(t.strip())
    if not m:
        return None
    sep, rest = m.group(3), m.group(4)
    if not sep and CAP_VERB.match(rest):
        return None
    return (m.group(1).rstrip(".").lower(), m.group(2), rest)

KNOWN = {"abstract", "introduction", "conclusion", "conclusions",
         "references", "acknowledgments", "acknowledgements", "background",
         "related work", "motivation", "methodology", "evaluation",
         "discussion", "appendix"}


# ---------------------------------------------------------------- page model

class Item:
    __slots__ = ("rect", "lines", "text", "own", "size", "med", "bold",
                 "kind", "level", "capt", "region", "group")

    def __init__(self, rect, lines, size, med, bold, text):
        self.rect = rect
        self.lines = lines          # list of (text, size, bold)
        self.text = text            # joined paragraph text
        # `text` grows when the next item turns out to continue this
        # paragraph; `own` stays the text of this item's own lines.  Whoever
        # lists the page's *lines* must read `own`: a paragraph continued
        # across a column break would otherwise print its tail twice, once
        # inside the head's line and once as the tail's own line.
        self.own = text
        self.size = size
        self.med = med
        self.bold = bold
        self.kind = "p"
        self.level = 0
        self.capt = None
        self.region = None
        self.group = [self]         # items absorbed into this one

    def __repr__(self):
        return f"<{self.kind} {self.rect.y0:.0f} {self.text[:30]!r}>"


def _joined(lines):
    """Text of span groups, with spaces reconstructed from page geometry."""
    out, x1, y1, y0 = "", None, None, None
    for t, sz, b, bb in lines:
        if out and x1 is not None:
            # a new line moves the *top* edge down; comparing against the
            # previous bottom fails on tight leading whose boxes overlap by a
            # couple of points (it glued "Department" to "University")
            # ...and the run must start left of where the last one ended: a
            # subscript's box also sits lower, but it does not wrap back
            newline = bb[1] > y0 + 1.0 and bb[0] < x1 - 1
            # a line that ends in a hyphen carries a trailing space as often
            # as not ("accesspat- " + "terns"), so test the trimmed tail
            if out.rstrip().endswith("-") and t.lstrip()[:1].islower():
                out = out.rstrip()[:-1]
                t = t.lstrip()
            elif not out.endswith(" ") and not t.startswith(" "):
                gap = bb[0] - x1
                # a word space is usually a 2-3pt gap, but tight kerning can
                # leave less than one; case change is a second signal
                if (newline or gap > 1.2
                        or (gap > .35 and out[-1].islower()
                            and t[:1].isupper())):
                    out += " "
        out += t
        x1, y1, y0 = bb[2], bb[3], bb[1]
    return out


def _cap_into_lines(cap, lines):
    """Put a drop cap back on the first line of the paragraph it opens.

    The cap is its own oversized one-letter run, so it is not one of the
    paragraph's lines.  The emitted HTML is rebuilt from the paragraph's
    lines, so the letter has to move into them or the first word loses it.
    """
    if not lines:
        return lines
    t, sz, b, bb = lines[0]
    return [(cap + t.lstrip(), sz, b, bb)] + list(lines[1:])


def page_items(page):
    """Text blocks of one page, split into paragraph-sized items."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        raw = []
        for l in b["lines"]:
            if l.get("dir") != (1.0, 0.0):
                continue                      # vertical stamps, rotated art
            # keep whitespace-only spans: in this journal's PDFs the spaces
            # between words are spans of their own
            spans = [sp for sp in l["spans"] if sp["text"]]
            if not any(sp["text"].strip() for sp in spans):
                continue
            # a heading often shares its line with the first words of the
            # paragraph ("1. Introduction Historically the rate ..."), so
            # split the line wherever the type size jumps
            first = next(sp for sp in spans if sp["text"].strip())
            groups, last = [[spans[0]]], first["size"]
            for sp in spans[1:]:
                sz = sp["size"] if sp["text"].strip() else last
                if abs(sz - last) > max(.45, .045 * last):
                    groups.append([sp])
                else:
                    groups[-1].append(sp)
                if sp["text"].strip():
                    last = sz
            for g in groups:
                txt, px1 = "", None
                for sp in g:
                    t = sp["text"]
                    if (txt and px1 is not None and not txt.endswith(" ")
                            and not t.startswith(" ")
                            and sp["bbox"][0] - px1 > 1.2):
                        txt += " "
                    txt += t
                    px1 = sp["bbox"][2]
                sized = [sp for sp in g if sp["text"].strip()]
                if not txt.strip() or not sized:
                    continue
                size = max(sp["size"] for sp in sized)
                bold = is_bold(" ".join(sp["font"] for sp in sized),
                               max(sp["flags"] for sp in sized))
                bb = pymupdf.Rect(g[0]["bbox"])
                for sp in g[1:]:
                    bb |= pymupdf.Rect(sp["bbox"])
                raw.append((txt, size, bold, tuple(bb)))
        if not raw:
            continue
        # split the block where the vertical gap is clearly larger than usual
        gaps = [raw[i + 1][3][1] - raw[i][3][3] for i in range(len(raw) - 1)]
        med_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
        groups, cur = [], [raw[0]]
        for i, ln in enumerate(raw[1:]):
            szjump = abs(ln[1] - cur[-1][1]) > max(.45, .045 * cur[-1][1])
            inline_head = (len(ln[0]) < 70 and re.match(
                r"^(?:\d{1,2}(?:\.\d{1,2})+|[IVX]{1,4}|[A-Z])\.\s+\S",
                ln[0].strip()))
            if (gaps[i] > max(6.0, med_gap * 1.45) or szjump or inline_head
                    or ln[0].lstrip()[:1] in "•▪◦"):
                groups.append(cur)
                cur = [ln]
            else:
                cur.append(ln)
        groups.append(cur)
        for g in groups:
            rect = pymupdf.Rect(g[0][3])
            for _, _, _, bb in g[1:]:
                rect |= pymupdf.Rect(bb)
            chars = sum(len(t) for t, _, _, _ in g)
            med = sum(s * len(t) for t, s, _, _ in g) / max(1, chars)
            out.append(Item(rect, g, max(s for _, s, _, _ in g), med,
                            all(b for _, _, b, _ in g), _joined(g)))
    return merge_fragments(out)


def merge_fragments(items):
    """Re-join a heading that small caps split into size groups.

    IEEE-style headings print the leading capital in one size and the rest in
    small caps, so "II. BACKGROUND AND MOTIVATION" arrives as four items on
    one baseline.  Only pieces of *different* sizes are merged: table cells
    side by side are the same size and stay separate.
    """
    items.sort(key=lambda i: (round((i.rect.y0 + i.rect.y1) / 2 / 5),
                              i.rect.x0))
    out, info = [], []          # info: (size of last piece, already merged?)
    for it in items:
        if out:
            p = out[-1]
            psz, mflag = info[-1]
            same_line = (abs((p.rect.y0 + p.rect.y1) / 2
                             - (it.rect.y0 + it.rect.y1) / 2) < 2.5
                         and 0 <= it.rect.x0 - p.rect.x1 < 14
                         and min(psz, it.size) > 6.5
                         and len(p.text) + len(it.text) < 70
                         and (abs(psz - it.size) > .5 or mflag))
            if same_line:
                p.lines = p.lines + it.lines
                p.text = p.own = _joined(p.lines)
                p.rect = p.rect | it.rect
                info[-1] = (it.size, True)
                continue
        out.append(it)
        info.append((it.size, False))
    return out


def body_size(items):
    hist = {}
    for it in items:
        for t, s, _, _ in it.lines:
            key = round(s * 2) / 2
            hist[key] = hist.get(key, 0) + len(t)
    return max(hist, key=hist.get) if hist else 10.0


def columns(items, textbox, body):
    """Return (left, right) x-ranges if the page is two-column, else None."""
    W = textbox.x1 - textbox.x0
    # a proportional band, not a fixed one: reference lists are set a point
    # smaller and can make the document-wide body size the smaller of the
    # two, which used to hide every full-size paragraph on the page and send
    # a two-column page down the single-column path
    cand = [i for i in items
            if body * .78 <= i.med <= body * 1.28
            and len(i.text) > 20
            # a stray one-word fragment (a page number caught in the band)
            # otherwise spoils the column edges and the page reads as
            # single-column, which interleaves a spanning abstract
            and .28 * W < i.rect.width < .58 * W]
    # a figure-heavy page can carry only one paragraph per column; the gutter
    # test below is what decides, not the count
    if len(cand) < 3:
        return None
    mid = (textbox.x0 + textbox.x1) / 2
    left = [i for i in cand if i.rect.x1 < mid + 10]
    right = [i for i in cand if i.rect.x0 > mid - 10]
    if not left or not right:
        return None
    lx1 = sorted(i.rect.x1 for i in left)[int(.9 * (len(left) - 1))]
    rx0 = sorted(i.rect.x0 for i in right)[int(.1 * len(right))]
    if lx1 > rx0 - 4:
        return None
    lx0 = sorted(i.rect.x0 for i in left)[0]
    rx1 = sorted(i.rect.x1 for i in right)[-1]
    return (lx0, lx1), (rx0, rx1)


def full_width(it, cols):
    return (cols is not None
            and it.rect.x0 < cols[0][1] - 8 and it.rect.x1 > cols[1][0] + 8)


def order_items(items, cols):
    if cols is None:
        return sorted(items, key=lambda i: (round(i.rect.y0), i.rect.x0))
    mid = (cols[0][1] + cols[1][0]) / 2

    def colkey(i):
        return (0 if i.rect.x0 < mid else 1, round(i.rect.y0), i.rect.x0)

    fulls = sorted((i for i in items if full_width(i, cols)),
                   key=lambda i: i.rect.y0)
    rest = [i for i in items if not full_width(i, cols)]
    out, y = [], -1e9
    for f in fulls:
        cy = (f.rect.y0 + f.rect.y1) / 2
        band = [i for i in rest
                if y <= (i.rect.y0 + i.rect.y1) / 2 < cy]
        out.extend(sorted(band, key=colkey))
        out.append(f)
        y = cy
    out.extend(sorted([i for i in rest
                       if (i.rect.y0 + i.rect.y1) / 2 >= y], key=colkey))
    return out


# ---------------------------------------------------------------- structure

def classify(items, body):
    for it in items:
        if it.kind == "drop":
            continue
        t = it.text.strip()
        low = t.lower().rstrip(".:")
        head = None
        m = HEAD_PAT.match(t)
        rm = None if m else ROMAN_PAT.match(t)
        lm = None if (m or rm) else LETTER_PAT.match(t)
        # headings: larger than body, or a numbered/bold short line.
        # IEEE prints "II. BACKGROUND" at body size and not bold, so roman
        # numerals and single letters are accepted on length alone.
        digits = sum(c.isdigit() for c in t) / max(1, len(t))
        if digits > .4:
            continue
        # a heading set larger than body still looks like a heading: it has
        # capitals and a word, and is not a figure's axis tick row
        bigger = (it.size > body * 1.12 and len(t) < 90
                  and re.search(r"[A-Z]", t) and re.search(r"[A-Za-z]{3}", t)
                  and sum(c.isdigit() for c in t) <= 2)
        # the word after the number must be a word: "1 0 1 0BID" is a figure
        # legend that happens to start with a digit
        numbered = (bool(m) and m.group(2)[:1].isalpha()
                    and it.size >= body * .98 and len(t) < 90
                    and (it.bold or it.size > body * 1.03))
        roman = bool(rm) and it.size >= body * .95 and len(t) < 90 and (
            len(t.split()) <= 9)
        letter = bool(lm) and it.size >= body * .95 and len(t) < 90 and (
            len(t.split()) <= 9 and len(t) > 4)
        named = low in KNOWN and len(t) < 60 and it.size > body * 1.0
        if (bigger or numbered or roman or letter or named) and len(t) > 1:
            head = t
        if head:
            it.kind = "h"
            if m:
                it.level = m.group(1).count(".") + 1   # 1 -> h2, 2 -> h3
                it.own = it.text = f"{m.group(1)} {m.group(2)}"
            elif rm:
                it.level = 1
                it.own = it.text = f"{rm.group(1)}. {rm.group(2)}"
            elif lm:
                it.level = 2
                it.own = it.text = f"{lm.group(1)}. {lm.group(2)}"
            elif low in KNOWN:
                it.level = 1
            else:
                it.level = 1 if it.size > body * 1.35 else 2
            continue
        cm = parse_caption(t)
        if cm and it.size < body * 1.05 and len(t) > 4:
            it.kind = "cap"
            it.capt = cm
            continue
        if low.startswith(("index terms", "keywords", "ccs concepts",
                           "categories and subject descriptors",
                           "general terms", "additional key words")):
            it.kind = "kw"
            continue
        if it.size < body * .86:
            it.kind = "small"


# ---------------------------------------------------------------- figures

def text_strip_rects(page, textbox):
    return textbox


def is_prose(it, colw, body):
    """Body text, as opposed to a chart label, a caption or a table cell."""
    if abs(it.med - body) > .28:
        return False
    if it.rect.width < .70 * colw:
        return False
    # a table row can be two full-width lines; real prose is also long
    return len(it.lines) >= 2 and len(it.text) >= 90


def graphic_rects(page):
    rs = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width > .6 and r.height > .6:
            rs.append(r)
    try:
        for im in page.get_image_info():
            rs.append(pymupdf.Rect(im["bbox"]))
    except Exception:
        pass
    return rs


def rule_boxes(page):
    """Clusters of zero-thickness drawings: tables and plots drawn with rules.

    `graphic_rects` keeps only rects with both dimensions, because a
    zero-height rule is not artwork for region growing.  A table whose cells
    are separated by `\\hline` has nothing else, so a renderer that crops
    from the graphics list finds no image at all and prints the caption
    alone.  The rules are clustered into one box per stack and kept aside,
    not mixed into `grects`, so the converter's own decisions do not move.
    """
    thin = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width <= .6 or r.height <= .6:
            thin.append((r.x0, r.y0, max(r.x1, r.x0 + .8),
                         max(r.y1, r.y0 + .8)))
    thin.sort(key=lambda r: (r[1], r[0]))
    boxes = []
    for r in thin:
        if boxes:
            b = boxes[-1]
            if (r[1] - b[3] < 20
                    and min(r[2], b[2]) - max(r[0], b[0]) > 0):
                boxes[-1] = (min(b[0], r[0]), min(b[1], r[1]),
                             max(b[2], r[2]), max(b[3], r[3]))
                continue
        boxes.append(r)
    # a lone rule is not artwork; a stack of them is a table or a plot
    return [pymupdf.Rect(b) for b in boxes
            if b[3] - b[1] > 12 and b[2] - b[0] > 40]


def _xover(r, x0, x1):
    ov = min(r.x1, x1) - max(r.x0, x0)
    return ov > max(1.0, .5 * min(r.width, x1 - x0))


def _artlike(it, colw, body):
    """Text that belongs to a figure: labels, legends, in-figure titles."""
    if it.kind in ("cap", "h", "drop"):
        return False
    return not is_prose(it, colw, body)


def _barrier(it, colw, body):
    """Something the figure region may not grow across."""
    return it.kind in ("cap", "h", "drop") or is_prose(it, colw, body)


def _tablish(it):
    """A table row or header set as text: no closing full stop, short cells.

    Old papers set tables as plain text with no rules at all, so nothing in
    the geometry marks them as artwork; the caption is the only anchor and
    the rows read as prose to is_prose.  Cross them only under a table
    caption, never under a figure caption.
    """
    t = it.text.strip()
    if not t or t.endswith((".", "!", "?")):
        return False
    return len(t.split()) <= 12 or any(c.isdigit() for c in t)


def _codeish(it):
    """A listing line rather than a chart label.

    These papers set code samples inside a drawn box, so the box looks like
    figure artwork; dropping everything short that sits in a graphic rect
    deleted the listings themselves.  Semicolons, braces, pointer arrows and
    call/subscript syntax say "code" without matching prose labels ("(b)
    PSHRs", "M1", "0.15").
    """
    t = it.text
    return (any(ch in t for ch in ";{}[]")
            or "->" in t or "++" in t
            or bool(re.search(r"[A-Za-z0-9_]\(", t)))


def _chain2(items, grects, y_start, x0, x1, body, upward, cap):
    """Grow a region from the caption over adjacent artwork.

    A chart's axis labels are text blocks, so "crop from the nearest text
    above" starts inside the figure.  Instead the region grows item by item
    while the gaps stay small, and it stops at the first piece of body text,
    heading or other caption -- which is what keeps a figure from swallowing
    the paragraph or the previous figure above it.
    """
    colw = x1 - x0
    y = y_start
    budget, saw, steps = 34, False, 0
    table = bool(cap.capt and cap.capt[0].startswith("t"))

    def crosses(it):
        """May the region step over this item?"""
        if it.kind in ("cap", "h", "drop"):
            return False
        # only table captions may step over unpunctuated prose-like runs;
        # under a figure caption the same test swallows real paragraphs
        return not is_prose(it, colw, body) or (table and _tablish(it))

    def barrier(it):
        return not crosses(it)

    while steps < 300:
        steps += 1
        if upward:
            lim = max([i.rect.y1 for i in items
                       if i is not cap and barrier(i)
                       and i.rect.y1 <= y + 1 and i.rect.x1 > x0
                       and i.rect.x0 < x1], default=-1e18)
            lo = max(y - budget, lim)
            cands, hasg = [], False
            for r in grects:
                if r.y1 <= y + 1 and r.y1 >= lo and _xover(r, x0, x1):
                    cands.append(r)
                    hasg = True
            for it in items:
                if it is cap or not crosses(it):
                    continue
                r = it.rect
                if r.y1 <= y + 1 and r.y1 >= lo and _xover(r, x0, x1):
                    cands.append(r)
            # a banner behind dropped text is one tall rect: cut it at the
            # barrier instead of letting its top edge drag the crop header-ward
            ny = max(min(r.y0 for r in cands), lim) if cands else y
        else:
            lim = min([i.rect.y0 for i in items
                       if i is not cap and barrier(i)
                       and i.rect.y0 >= y - 1 and i.rect.x1 > x0
                       and i.rect.x0 < x1], default=1e18)
            hi = min(y + budget, lim)
            cands, hasg = [], False
            for r in grects:
                if r.y0 >= y - 1 and r.y0 <= hi and _xover(r, x0, x1):
                    cands.append(r)
                    hasg = True
            for it in items:
                if it is cap or not crosses(it):
                    continue
                r = it.rect
                if r.y0 >= y - 1 and r.y0 <= hi and _xover(r, x0, x1):
                    cands.append(r)
            ny = min(max(r.y1 for r in cands), lim) if cands else y
        if not cands or abs(ny - y) < .5:
            break
        y, budget, saw = ny, 18, saw or hasg
    return y, saw


def _contentful(rect, items, grects, cap):
    """Does this crop actually contain artwork or tabular text?"""
    if rect is None:
        return False
    n = 0
    for i in items:
        if i is cap or not i.text.strip():
            continue
        p = pymupdf.Point((i.rect.x0 + i.rect.x1) / 2,
                          (i.rect.y0 + i.rect.y1) / 2)
        if rect.contains(p) and i.rect.width > 20:
            n += 1
    if n >= 2:
        return True
    area = 0.0
    for g in grects:
        ov = pymupdf.Rect(g) & rect
        if ov.width > 2 and ov.height > 2:
            area += ov.width * ov.height
    return area > .05 * rect.width * rect.height


def find_region(cap, items, cols, textbox, body, grects, frame):
    """The page rectangle that belongs to one caption, or None."""
    x0, x1 = textbox.x0, textbox.x1
    if cols:
        mid = (cols[0][1] + cols[1][0]) / 2
        cw = (cap.rect.x0 + cap.rect.x1) / 2
        if cw < mid:
            x0, x1 = cols[0]
        elif cw > mid:
            x0, x1 = cols[1]
        if cap.rect.x0 < cols[0][1] - 8 and cap.rect.x1 > cols[1][0] + 8:
            x0, x1 = textbox.x0, textbox.x1
    colw = x1 - x0
    kind, num, _ = cap.capt

    top, saw = _chain2(items, grects, cap.rect.y0, x0, x1, body, True, cap)
    up = None
    if cap.rect.y0 - top > 10:
        up = pymupdf.Rect(x0, top - 2, x1, cap.rect.y0 - 2)

    if kind.startswith("fig"):
        down = None
        if up is None or cap.rect.y0 - top < 60:
            bot2, saw2 = _chain2(items, grects, cap.rect.y1, x0, x1, body,
                                 False, cap)
            if saw2 and bot2 - cap.rect.y1 > 10:
                down = pymupdf.Rect(x0, cap.rect.y1 + 1, x1, bot2 + 1)
        cands = [up, down]
        # caption close to its artwork: whichever side is taller wins
        if (up and down and (not saw or cap.rect.y0 - top < 60)
                and down.height > up.height):
            cands.reverse()
    else:
        bot, saw = _chain2(items, grects, cap.rect.y1, x0, x1, body, False, cap)
        below = [i for i in items if i is not cap
                 and i.rect.y0 >= cap.rect.y1 - 1
                 and i.rect.x1 > x0 and i.rect.x0 < x1
                 and i.rect.y1 <= bot + 2]
        down = None
        if bot - cap.rect.y1 > 10 and (saw or _tabley(below, colw)):
            down = pymupdf.Rect(x0, cap.rect.y1 + 1, x1, bot + 1)
        cands = [down, up]
        # a caption can be set below its grid; pick the side that actually
        # holds table rows instead of trusting the usual caption-above order
        if down and up:
            def nrows(r):
                return sum(1 for i in items if i is not cap
                           and r.contains(pymupdf.Point(
                               (i.rect.x0 + i.rect.x1) / 2,
                               (i.rect.y0 + i.rect.y1) / 2))
                           and _tablish(i))
            if nrows(up) > nrows(down):
                cands.reverse()

    rect = None
    for c in cands:
        if c is not None and _contentful(c, items, grects, cap):
            rect = c
            break
    if rect is not None:
        # never grow into the page header/footer: running heads and the
        # conference banner live there as vector art, so no text barrier
        # stops the chain and a table crop ends up framed by the banner
        rect = pymupdf.Rect(max(rect.x0, frame.x0), max(rect.y0, frame.y0),
                            min(rect.x1, frame.x1), min(rect.y1, frame.y1))
        if rect.height < 9 or rect.width < 9:
            rect = None
    return rect


def render_region(page, rect, out_png, dpi):
    if rect is None:
        return None
    H, W = page.rect.height, page.rect.width
    rect = pymupdf.Rect(max(rect.x0, 0), max(rect.y0, 0),
                        min(rect.x1, W), min(rect.y1, H))
    if rect.height < 12 or rect.width < 20:
        return None
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                          clip=rect, alpha=False)
    if pix.width > 1500:                  # keep pages light
        scale = 1500 / pix.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi * scale / 72,
                                                    dpi * scale / 72),
                              clip=rect, alpha=False)
    pix.save(str(out_png))
    return out_png


def _tabley(items, width):
    """True if these blocks look like tabular text rather than prose."""
    if not items:
        return False
    nlines, short = 0, 0
    for it in items:
        nlines += len(it.lines)
        w = it.rect.width
        if w < .8 * width:
            short += len(it.lines)
    prose_end = any(it.text.rstrip().endswith((".", "?", "!"))
                    and len(it.lines) > 2 and it.rect.width > .9 * width
                    for it in items)
    return nlines >= 3 and short >= .6 * nlines and not prose_end


# ---------------------------------------------------------------- document

def slugify(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    return re.sub(r"-+", "-", t)[:52].strip("-") or "paper"


def join_auth(its):
    """Author/affiliation lines, keeping comma-continued names on one line."""
    out = ""
    for i in its:
        t = i if isinstance(i, str) else i.text
        t = t.strip()
        if not out:
            out = t.lstrip(", ")
        elif t[:1] in ",;":
            out += t
        elif re.match(r"^(and|&)\b", t):
            out += " " + t
        else:
            out += " · " + t.lstrip(", ")
    return out


AFFIL = re.compile(r"(univers|institut|department|dept\.|laborator|\blab\b|school|"
                   r"college|corporation|corp\.|\binc\b|\bltd\b|\bgmbh\b|research|"
                   r"google|microsoft|intel|nvidia|\bamd\b|\barm\b|\bibm\b|csail|"
                   # the state/ZIP tail must stay case-sensitive, or ", Lu Li"
                   # matches it and a name lands in the affiliation line
                   r"cent(er|re)|faculty|academy|@|\{|\}|"
                   r"(?-i:,\s*[A-Z]{2}(?![A-Za-z])))", re.I)


def front_matter(p0, page_h, body):
    """Title, authors, affiliation, and the ids of the items they came from.

    Conservative on purpose: everything below the title is walked only while
    it is short and near body size, so a paper without an "Abstract" heading
    cannot have its first paragraph mistaken for the author list.
    """
    items = sorted([i for i in p0 if i.rect.y1 < .62 * page_h
                    and i.size >= body * .9], key=lambda i: i.rect.y0)
    if not items:
        return None
    maxsz = max(i.size for i in items)
    title = [i for i in items if i.size > maxsz - .6]
    if not title:
        return None
    ty0 = min(i.rect.y0 for i in title)
    ty1 = max(i.rect.y1 for i in title)
    # by top edge, not bottom: an author line set to the right of a short title
    # starts above the title's baseline box and would otherwise be skipped
    rest = sorted([i for i in items if i not in title and i.rect.y0 > ty0],
                  key=lambda i: (round(i.rect.y0 / 2), i.rect.x0))
    keep = []
    lastbottom = ty1
    for it in rest:
        t = it.text.strip()
        # the author block is one contiguous band; text further down the page
        # is figure lettering ("Contributions / Software / Hardware")
        if it.rect.y0 - lastbottom > 32:
            break
        if len(t) > 200 or it.size < body * .88:
            break
        if re.match(r"^\d{1,2}\.?\s+\S", t):
            break
        if norm(t).replace(".", " ").split()[0] in ("abstract", "introduction"):
            break
        keep.append(it)
        lastbottom = max(lastbottom, it.rect.y1)
        # author blocks print one item per name, so a four-line list easily
        # passes eight; walk further and let the stop conditions above decide
        if len(keep) >= 40:
            break
    auth, affil = [], []
    if keep:
        asz = max(i.size for i in keep)
        # authors and affiliations often share one type size, and an author
        # list can wrap onto a second line, so rows alone do not separate
        # them: institution words, e-mails and address braces do
        for i in keep:
            t = i.text.strip()
            m = AFFIL.search(t)
            if m and len(keep) == 1 and m.start() > 0:
                # names and affiliation printed as a single text run
                # (Triangel): the affiliation begins at the first institution
                # word. Only for a lone run — on a normal page "Ghent
                # University" is one whole affiliation item, not a name plus
                # an affiliation.
                auth.append(t[:m.start()].strip(" ,;·"))
                affil.append(t[m.start():].strip())
            elif (i.size <= asz - .6 or m or re.fullmatch(
                    r"[a-z][\w.]*(?:,\s?[a-z][\w.]*)+", t)):
                affil.append(t)
            else:
                auth.append(t)
    used = {id(i) for i in title + keep}
    return dict(title=" ".join(join_lines([t for t, _, _, _ in i.lines])
                               for i in title),
                authors=join_auth(auth), affil=join_auth(affil),
                used=used, y1=max([i.rect.y1 for i in title + keep]))


def running_heads(all_pages):
    """Text that repeats at the top/bottom of most pages (drop it)."""
    freq = {}
    for items, page in all_pages:
        seen = set()
        for it in items:
            # a header set under a banner sits well below y=70, so scan a
            # fraction of the page rather than a fixed band
            if (it.rect.y1 < .14 * page.rect.height
                    or it.rect.y0 > .86 * page.rect.height):
                # a multi-line block is a paragraph that merely sits in the
                # band, not a running head.  Count printed lines by height:
                # this journal encodes every word group as its own "line", so
                # len(it.lines) called a one-line footer four lines long.
                if it.rect.height > it.size * 2.4:
                    continue
                k = norm(it.text)[:80]
                if k and k not in seen:
                    freq[k] = freq.get(k, 0) + 1
                    seen.add(k)
    n = max(1, len(all_pages))
    # 35%: journals alternate two running heads (recto/verso), so each one
    # only covers about half the pages
    return {k for k, c in freq.items() if c >= max(3, n * .35)}


def convert(pdf, outroot, dpi=200):
    doc = pymupdf.open(pdf)
    npages = doc.page_count
    raw_pages = [(page_items(p), p) for p in doc]
    body = body_size([i for items, _ in raw_pages for i in items])
    heads = running_heads(raw_pages)

    # ---- drop furniture, tag structure
    pages = []
    for items, page in raw_pages:
        # enumerated list bullets print as a letter plus an enclosing circle
        # ("a⃝"), but extraction splits them: the letter ends up as its own
        # run at the close of the previous line, the circle opens the next
        # paragraph.  Put the label back before anything is dropped.
        circled = [i for i in items if i.text.startswith("⃝")]
        if circled:
            for it in items:
                t = it.text.strip()
                if len(t) != 1 or not t.isalpha() or it.size < 6.5:
                    continue
                cy = (it.rect.y0 + it.rect.y1) / 2
                cand = [c for c in circled
                        if c is not it and c.rect.y0 - 1 <= cy
                        <= c.rect.y0 + 14]
                if cand:
                    c = min(cand, key=lambda c: abs(c.rect.y0 - it.rect.y0))
                    c.own = c.text = t + c.text
                    it.kind = "drop"
        keep = []
        for it in items:
            k = norm(it.text)[:80]
            # running heads, page numbers and arXiv stamps stay in the list
            # as "drop": they are not printed, but a figure crop must not
            # grow across them either
            # a journal often sets the same running head as two text runs on
            # one page and one run on the next; the merged form still counts
            if k in heads or any(h in k for h in heads if len(h) > 8):
                it.kind = "drop"
            elif re.fullmatch(r"\d{1,3}", k) and (
                    it.rect.y1 < .14 * page.rect.height
                    or it.rect.y0 > .86 * page.rect.height):
                it.kind = "drop"
            elif re.match(r"^arxiv:\S+\s+\[", k) or k.startswith("arxiv:"):
                it.kind = "drop"
            # footnote daggers / asterisks set as their own text run
            elif len(k) <= 3 and not any(c.isalnum() for c in k):
                it.kind = "drop"
            # single letters are figure-panel or algorithm-step labels; kept
            # out of the prose.  A drop cap is the one exception: it is set
            # at display size and the merge pass stitches it to its word
            elif (len(k) == 1 and k.isalpha()
                  and it.size <= body * 1.4):
                it.kind = "drop"
            elif re.match(r"^(to appear at|appears in the|in proceedings\b|"
                          r"proceedings of)\b", k):
                it.kind = "drop"
            # publisher copyright notice: the same small print on page one of
            # every ACM/IEEE paper.  It is not part of the work.  Match the
            # fragments that survive OCR ("permlssnon", "copes").
            elif ("make digital or hard cop" in k
                  or k.startswith(("personal or classroom use",
                                   "to copy otherwise",
                                   "requires prior speci"))):
                it.kind = "drop"
            keep.append(it)
        # the text area is the union of real text, not just the runs set at
        # the document-wide body size: a reference page is a point smaller and
        # used to collapse the box to the other column, which then made the
        # page look single-column
        tb = pymupdf.Rect(1e9, 1e9, -1e9, -1e9)
        for it in keep:
            if (it.kind != "drop" and it.med >= body * .75
                    and len(it.text) > 20):
                tb |= it.rect
        if tb.is_empty:
            tb = pymupdf.Rect(40, 40, page.rect.width - 40,
                              page.rect.height - 40)
        live = [i for i in keep if i.kind != "drop"]
        pages.append(dict(items=keep, page=page, cols=columns(live, tb, body),
                          tb=tb, grects=graphic_rects(page),
                          lrects=rule_boxes(page),
                          # the document is closed before the caller sees
                          # these dicts, so the page geometry has to be kept
                          # here rather than read back off the Page object
                          size=(page.rect.width, page.rect.height),
                          # image rectangles are not in get_drawings(); a
                          # label inside a raster figure is otherwise
                          # indistinguishable from body text
                          imgs=[r for x in page.get_images(full=True)
                                for r in page.get_image_rects(x[0])]))

    fm = front_matter(pages[0]["items"], pages[0]["page"].rect.height,
                      body)
    if fm:
        pages[0]["items"] = [i for i in pages[0]["items"]
                             if id(i) not in fm["used"]]
    # Whatever still sits above the abstract on page one is front matter the
    # author block did not claim: e-mail scraps, "USA", conference lines. It
    # is not prose, and left in it becomes a drop-capped opening paragraph.
    p0 = pages[0]
    ordered0 = order_items(p0["items"], p0["cols"])
    ai = next((n for n, it in enumerate(ordered0)
               if norm(it.text).startswith("abstract")), None)
    if ai is not None:
        for it in ordered0[:ai]:
            if it.size < body * 1.5 and not CAP_PAT.match(it.text.strip()):
                it.kind = "drop"
    for pd in pages:
        classify(order_items(pd["items"], pd["cols"]), body)
        # some journals print "Figure 3" on its own line, with the caption
        # text in the block underneath
        for it in pd["items"]:
            if it.kind == "cap" and not it.capt[2].strip():
                nxt = [j for j in pd["items"] if j is not it
                       and j.rect.y0 >= it.rect.y1 - 2
                       and j.rect.x1 > it.rect.x0 and j.rect.x0 < it.rect.x1]
                if nxt:
                    j = min(nxt, key=lambda k: k.rect.y0)
                    if (j.rect.y0 - it.rect.y1 < 14
                            and j.size <= body * 1.05
                            and j.kind not in ("cap", "h")
                            and len(j.text) > 12):
                        it.capt = (it.capt[0], it.capt[1], j.text)
                        j.kind = "drop"

    # figure regions first, so that the labels inside them can be removed from
    # the text flow before anything is emitted
    for pd in pages:
        for it in pd["items"]:
            if it.kind == "cap":
                pr = pd["page"].rect
                frame = pymupdf.Rect(pr.x0 + 3, pr.y0 + .07 * pr.height,
                                     pr.x1 - 3, pr.y0 + .94 * pr.height)
                it.region = find_region(it, pd["items"], pd["cols"], pd["tb"],
                                        body, pd["grects"], frame)
        # two captions facing each other (a table above, its figure below)
        # can lay claim to the same band; split it halfway between them
        caps = sorted([i for i in pd["items"]
                       if i.kind == "cap" and i.region],
                      key=lambda i: i.rect.y0)
        # (every pair, not neighbours: a right-column caption can sit between
        #  two left-column ones, which is exactly where the overlap happens)
        for ai, a in enumerate(caps):
            for b in caps[ai + 1:]:
                ra, rb = a.region, b.region
                if ra.y1 <= rb.y0 + 8 or rb.y1 <= ra.y0 + 8:
                    continue
                if ra.x1 <= rb.x0 or ra.x0 >= rb.x1:
                    continue
                mid = (a.rect.y1 + b.rect.y0) / 2
                if ra.y0 < mid < ra.y1:
                    ra.y1 = mid
                if rb.y0 < mid < rb.y1:
                    rb.y0 = mid
        regs = [i.region for i in pd["items"] if i.kind == "cap" and i.region]
        page_area = pd["page"].rect.width * pd["page"].rect.height
        for it in pd["items"]:
            if it.kind in ("p", "small", "kw", "h"):
                c = pymupdf.Point((it.rect.x0 + it.rect.x1) / 2,
                                  (it.rect.y0 + it.rect.y1) / 2)
                if any(r.contains(c) for r in regs):
                    it.kind = "drop"
                # chart legends and axis labels often sit just outside the
                # crop the caption claimed: "Memory latency Compute time",
                # "BDSCH MemLat=70".  A label is short, unpunctuated, and
                # sits over the artwork -- unlike a body paragraph, which is
                # long and closes a sentence.
                elif (it.kind in ("p", "small", "kw")
                      and len(it.text.split()) <= 10
                      and not it.text.strip().endswith((".", "?", "!", ":"))
                      and not _codeish(it)
                      and any(g.contains(c) for g in pd["grects"])):
                    it.kind = "drop"
                # chart legends and axis labels often sit just outside the
                # crop the caption claimed: "Memory latency Compute time",
                # "BDSCH MemLat=70".  A label is short, unpunctuated, and
                # sits over the artwork -- unlike a body paragraph, which is
                # long and closes a sentence.
                elif (it.kind in ("p", "small", "kw")
                      and len(it.text.split()) <= 10
                      and not it.text.strip().endswith((".", "?", "!"))
                      and any(r.x0 < it.rect.x1 and r.x1 > it.rect.x0
                              and it.rect.y0 < r.y1 + 40
                              and it.rect.y1 > r.y0 - 40 for r in regs)):
                    it.kind = "drop"

    slug = slugify(fm["title"] if fm else Path(pdf).stem)
    outdir = outroot / slug
    outdir.mkdir(parents=True, exist_ok=True)
    assets = outdir / "assets"

    # ---- reading order for the whole document, with paragraph repair
    # (a sentence split across a column or page break loses its tail word
    #  unless the halves are stitched back together)
    seq = []
    for pd in pages:
        for it in order_items(pd["items"], pd["cols"]):
            # dropped furniture (page numbers, running heads) must not sit
            # between a paragraph's two column halves and stop the stitch
            if it.kind == "drop":
                continue
            seq.append((pd, it))
    merged = []
    pending_cap = ""
    for pd, it in seq:
        # a drop cap is its own oversized one-letter text run; depending on
        # the sort it lands just before or just after the paragraph it opens
        cap = it.text.strip()
        if (it.kind == "p" and len(cap) == 1 and cap.isalpha()
                and it.size > body * 1.4):
            if (merged and merged[-1][1].kind == "p"
                    and merged[-1][1].text[:1].islower()):
                prev = merged[-1][1]
                prev.text = cap + prev.text
                prev.lines = _cap_into_lines(cap, prev.lines)
                prev.group.append(it)
            else:
                pending_cap = cap
            continue
        if pending_cap and it.kind == "p" and it.text.lstrip()[:1].islower():
            # the cap's own line disappears into this paragraph, so the
            # paragraph's line does carry it
            it.own = it.text = pending_cap + it.text.lstrip()
            it.lines = _cap_into_lines(pending_cap, it.lines)
            pending_cap = ""
        if merged and it.kind == "p" and merged[-1][1].kind == "p":
            prev = merged[-1][1]
            pv = prev.text
            # continuation items keep the leading space of their print line
            nt = it.text.lstrip()
            # a span set in another font (typewriter, italic) can split a
            # sentence so the closing bracket or full stop opens the next
            # run; it belongs to the previous sentence, not a new paragraph
            if nt[:1] in ")]}”\"'.,;:!?" and pv and pv[-1] not in ".!?":
                prev.text = pv.rstrip() + nt
                prev.group.append(it)
                continue
            if pv.endswith("-") and nt[:1].islower():
                prev.text = pv[:-1] + nt
                prev.group.append(it)
                continue
            if pv and pv[-1] not in ".!?:;”\"')]}—" and nt[:1].islower():
                prev.text = pv + " " + nt
                prev.group.append(it)
                continue
        merged.append((pd, it))

    # ---- emit body
    flow, toc, fig_no = [], [], 0
    # the drop cap belongs on the first body paragraph after the abstract,
    # as in the template, not on a title-page scrap and not on the abstract
    opened = False
    in_abs = False
    sec, sub = 0, 0
    open_sec = False

    def close_sec():
        nonlocal open_sec, in_abs
        if open_sec:
            flow.append("</section>")
            open_sec = False
            in_abs = False

    for pd, it in merged:
        if it.kind == "drop":
            continue
        if it.kind == "cap":
            fig_no += 1
            kind, num, rest = it.capt
            isfig = kind.startswith("fig")
            label = ("Figure" if isfig else "Table") + f" {num}."
            png = render_region(pd["page"], it.region,
                                assets / f"f{fig_no:02d}.png", dpi)
            cap = (f'<figcaption><span class="capnum">{esc(label)}</span>'
                   f'{esc(rest)}</figcaption>')
            if not png:
                flow.append(f'<p class="rawcap"><span class="capnum">'
                            f'{esc(label)}</span>{esc(rest)}</p>')
            elif isfig:
                flow.append(f'<figure class="plate plate--col">\n'
                            f'  <div class="plate__frame"><img '
                            f'src="assets/{png.name}" '
                            f'alt="{esc(rest)[:160]}"></div>\n  {cap}\n'
                            f'</figure>')
            else:
                flow.append(f'<figure class="table">\n'
                            f'  <div class="tablewrap"><img '
                            f'src="assets/{png.name}" '
                            f'alt="{esc(rest)[:160]}"></div>\n  {cap}\n'
                            f'</figure>')
            continue
        if it.kind == "h":
            if it.level <= 1:
                text = it.text
                if norm(text).startswith("abstract") and not any(
                        t[1] == "abstract" for t in toc):
                    close_sec()
                    flow.append('<section id="abstract" class="abstract">'
                                '<h2>Abstract</h2>')
                    open_sec = True
                    in_abs = True
                    toc.append((1, "abstract", "Abstract"))
                    continue
                close_sec()
                sec += 1
                sub = 0
                cur_id = f"s{sec}"
                m = re.match(r"^(\d+)\s+(.*)$", text)
                flow.append(f'<section id="{cur_id}">')
                open_sec = True
                if m:
                    flow.append(f'<h2 class="sec"><span class="n">'
                                f'{m.group(1)}</span>{esc(m.group(2))}</h2>')
                    toc.append((1, cur_id, m.group(0)))
                else:
                    flow.append(f'<h2 class="sec">{esc(text)}</h2>')
                    toc.append((1, cur_id, text))
            else:
                sub += 1
                cur_id = f"s{sec}-{sub}"
                flow.append(f'<h3 class="sub" id="{cur_id}">'
                            f'{esc(it.text)}</h3>')
                toc.append((2, cur_id, it.text))
            continue
        # a standalone "Abstract" label set as a paragraph rather than a
        # heading: the NeurIPS/ICLR templates and several IEEE styles put the
        # label and its body in separate blocks, so the heading branch above
        # never fires and the abstract is rendered as an ordinary opening
        # paragraph.  Page 1 only; a survey's own table of contents may print
        # the word on its own line later in the document.
        if (not in_abs and pd["page"].number <= 1
                and norm(it.text).strip(" :.") == "abstract"
                and not any(t[1] == "abstract" for t in toc)):
            close_sec()
            flow.append('<section id="abstract" class="abstract">'
                        '<h2>Abstract</h2>')
            open_sec = True
            in_abs = True
            toc.append((1, "abstract", "Abstract"))
            continue
        if it.kind == "kw":
            t = it.text.strip()
            # ACM sets the label on its own line and the list under it; the
            # list's block can start a point higher, so reading order emits
            # the list first and the label lands below the thing it names
            if (norm(t) in ("ccs concepts", "keywords")
                    and flow and flow[-1].startswith("<p>• ")):
                block = flow.pop()
                flow.append(f'<div class="kw"><p>{esc(t)}</p></div>')
                flow.append(block)
                continue
            # label and value on one line: without the bold face the two
            # run together ("Keywords Software Prefetching"), so put back a
            # separator
            m2 = re.match(r"^(categories and subject descriptors|ccs concepts|"
                          r"additional key words|index terms|keywords|"
                          r"general terms)\s+(?=\S)", t, re.I)
            if m2:
                t = t[:m2.end()].strip() + ": " + t[m2.end():]
            flow.append(f'<div class="kw"><p>{esc(t)}</p></div>')
            continue
        if it.kind == "small":
            flow.append(f'<p class="fnl">{esc(it.text)}</p>')
            continue
        # "Abstract—The memory wall …" printed as one paragraph (IEEE style):
        # split it so the abstract gets its section and the drop cap does not
        # land on the "A" of "Abstract"
        am = re.match(r"^(abstract)\s*[—–―:.\-]\s*(.+)$",
                      it.text.strip(), re.I | re.S)
        if am and it.kind == "p" and not any(t[1] == "abstract" for t in toc):
            close_sec()
            flow.append('<section id="abstract" class="abstract">'
                        '<h2>Abstract</h2>')
            open_sec = True
            in_abs = True
            toc.append((1, "abstract", "Abstract"))
            flow.append(f'<p>{esc(am.group(2))}</p>')
            continue
        cls = "" if opened or in_abs else ' class="opening"'
        flow.append(f'<p{cls}>{esc(it.text)}</p>')
        opened = opened or not in_abs
    close_sec()
    doc.close()
    return dict(slug=slug, flow=flow, toc=toc, fm=fm, npages=npages,
                body=body, outdir=outdir, doc=doc, pages=pages, items=merged)


# ---------------------------------------------------------------- html out

EXTRA_CSS = """
<style>
/* ---- additions for the batch reflow pages ---- */
.rawcap {
  font-family: var(--sans); font-size: 13px; line-height: 1.55;
  color: var(--ink-soft); border-left: 2px solid var(--hairline-strong);
  padding-left: .9rem; margin: 1.5rem 0;
}
.rawcap .capnum, .capnum { color: var(--accent); font-family: var(--mono);
  font-weight: 700; margin-right: .4rem; }
p.fnl { font-size: 15.5px; line-height: 1.62; color: var(--ink-soft);
  margin: 0 0 .8em; }
.tablewrap {
  border: 1px solid var(--hairline-strong); background: var(--plate);
  padding: .35rem; overflow-x: auto; line-height: 0;
}
.tablewrap img { display: block; margin: 0 auto; max-width: 100%; height: auto;
  filter: invert(1) hue-rotate(180deg) brightness(.96) contrast(.94); }
figure.table { margin: 1.8rem 0 2rem; }
.mast__note { font-family: var(--mono); font-size: 11.5px; letter-spacing: .04em;
  color: var(--ink-faint); text-transform: uppercase; margin: .9rem 0 0; }
/* bibliography URLs offer no break opportunities (slashes, commas, dots) and
   used to push the document 40px wide inside a 308px viewport */
.paper p, .paper li, .paper h1, .paper h2, .paper h3,
.paper figcaption, .paper td, .paper th, .paper code, .paper a {
  overflow-wrap: anywhere;
}
</style>
"""


def head_html(title):
    txt = FDP.read_text(encoding="utf-8")
    # from <head>, not from byte 0: the template's own doctype/<html> would
    # otherwise be emitted a second time above the one write_page opens with
    head = txt[txt.index("<head>"):txt.index("</style>")]
    head = re.sub(r"<title>.*?</title>", "<title>" + esc(title) + "</title>",
                  head, count=1, flags=re.S)
    return head + EXTRA_CSS + "</style>\n</head>\n<body>\n"


def script_html():
    txt = FDP.read_text(encoding="utf-8")
    i = txt.index("<script>")
    j = txt.index("</script>", i) + len("</script>")
    return txt[i:j]


def toc_html(toc):
    if not toc:
        return ""
    out = ["<ol>"]
    for lvl, sid, label in toc[:120]:
        out.append(f'<li><a href="#{sid}"'
                   + (' class="toc__sub"' if lvl == 2 else "")
                   + f'>{esc(label[:110])}</a></li>')
    out.append("</ol>")
    return "\n".join(out)


def write_page(res, outroot, src_name):
    fm = res["fm"] or {}
    title = fm.get("title") or Path(src_name).stem
    short = re.sub(r":.*$", "", title).strip() or title
    if len(short) > 72:
        short = short[:69].rstrip() + "…"
    toc = res["toc"]
    body = "\n\n      ".join(res["flow"])
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        head_html(short + " · " + "reading edition"),
        '<div class="toprule"></div>',
        "",
        '<div class="shell">',
        '  <aside class="side">',
        '    <div class="toc">',
        '      <p class="toc__label">Contents</p>',
        '      <button class="toc-toggle" type="button" aria-expanded="false">'
        'Contents ▾</button>',
        '      <nav id="toc">',
        toc_html(toc),
        '      </nav>',
        '      <p class="toc__aside"><strong>Source</strong>'
        + esc(src_name) + '</p>',
        '    </div>',
        '  </aside>',
        '',
        '  <main class="paper">',
        '    <header class="mast">',
        f'      <p class="mast__kicker">Re-typeset from PDF · '
        f'{res["npages"]} pages</p>',
        f'      <h1 class="title">{esc(title)}</h1>',
    ]
    if fm.get("authors"):
        parts.append(f'      <p class="authors">{esc(fm["authors"])}</p>')
    if fm.get("affil"):
        parts.append(f'      <p class="affil">{esc(fm["affil"])}</p>')
    parts += [
        '      <p class="venue">Text reflowed from the two-column PDF; '
        'figures and tables are crops of the original pages. '
        'Section numbering follows the source.</p>',
        '    </header>',
        '',
        f'      {body}',
        '',
        '    <footer>',
        f'      <p>Source file <code>{esc(src_name)}</code>. '
        'This page is an automatic reading edition: the text was extracted '
        'from the PDF and reflowed into one column, and every figure is a '
        'crop of the page it appears on. Typographic detail, page breaks and '
        'the original pagination are not reproduced.</p>',
        '    </footer>',
        '  </main>',
        '</div>',
        '',
        script_html(),
        '</body>',
        '</html>',
    ]
    out = res["outdir"] / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def write_index(rows, outroot):
    cards = []
    for r in rows:
        cards.append(
            f'<a class="card" href="{r["slug"]}/index.html">\n'
            f'  <span class="card__n">{r["n"]:02d}</span>\n'
            f'  <span class="card__t">{esc(r["title"])}</span>\n'
            f'  <span class="card__m">{r["npages"]} pages · '
            f'{r["nfig"]} figures · {esc(r["src"])}</span>\n'
            f'</a>')
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reading list · re-typeset papers</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {{
    color-scheme: dark;
    --bg:#06070a; --paper:#0e1118; --plate:#05060a;
    --ink:#e9ebf2; --ink-soft:#a4a9ba; --ink-faint:#6a7080;
    --hairline:#1d212b; --hairline-strong:#2f3542;
    --accent:#8aa4ff; --accent-ink:#bdcaff; --accent-wash:#151b2c;
    --serif:"Newsreader",Georgia,serif;
    --sans:"Space Grotesk",system-ui,sans-serif;
    --mono:"JetBrains Mono",ui-monospace,monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:var(--serif); font-size:18px; line-height:1.6;
    -webkit-font-smoothing:antialiased; }}
  .toprule {{ height:3px; background:linear-gradient(90deg,var(--accent),#3c4a7a 60%,transparent); }}
  .wrap {{ max-width: 62rem; margin:0 auto; padding: 3.2rem 1.5rem 5rem; }}
  h1 {{ font-family:var(--serif); font-size:clamp(2rem,5vw,3rem);
    line-height:1.1; font-weight:500; letter-spacing:-.02em; margin:.6rem 0 .8rem; }}
  .kicker {{ font-family:var(--mono); font-size:11.5px; letter-spacing:.12em;
    text-transform:uppercase; color:var(--ink-faint); margin:0; }}
  .lede {{ font-family:var(--sans); color:var(--ink-soft); font-size:15px;
    max-width: 44rem; margin: 0 0 2.8rem; }}
  .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 1px;
    background: var(--hairline); border:1px solid var(--hairline); }}
  .card {{ display:flex; flex-direction:column; gap:.45rem; padding:1.15rem 1.2rem;
    background:var(--paper); text-decoration:none; color:inherit;
    transition: background .15s; min-width:0; }}
  .card:hover {{ background: var(--accent-wash); }}
  .card__n {{ font-family:var(--mono); font-size:11px; color:var(--ink-faint); }}
  .card__t {{ font-size:17.5px; line-height:1.3; }}
  .card:hover .card__t {{ color:var(--accent-ink); }}
  .card__m {{ font-family:var(--mono); font-size:10.5px; color:var(--ink-faint);
    margin-top:auto; overflow-wrap:anywhere; }}
  footer {{ margin-top:3rem; font-family:var(--sans); font-size:13px;
    color:var(--ink-faint); }}
  @media (max-width: 760px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="toprule"></div>
<div class="wrap">
  <p class="kicker">{len(rows)} papers · automatic reflow</p>
  <h1>Reading list</h1>
  <p class="lede">Every paper in <code>books/reading</code>, re-typeset as a
  single-column dark page: body text extracted and reflowed, figures and tables
  cropped from the original pages at 200&nbsp;dpi, a contents list generated
  from the paper's own section headings. These are reading editions, not
  facsimiles.</p>
  <div class="grid">
    {chr(10).join("    " + c.replace(chr(10), chr(10) + "    ") for c in cards)}
  </div>
  <footer>Built from the PDFs in <code>Desktop/books/reading</code>.
  Figures are page crops, so their typography is the publisher's, not this page's.</footer>
</div>
</body>
</html>
"""
    (outroot / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/c/Users/24514/Desktop/books/reading")
    ap.add_argument("--out", default="/home/trunk_trick/papers_html")
    ap.add_argument("--only", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    src, outroot = Path(a.src), Path(a.out)
    outroot.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in src.glob("*.pdf"))
    if a.only:
        pdfs = [p for p in pdfs if a.only.lower() in p.name.lower()]
    rows, used = [], set()
    for n, pdf in enumerate(pdfs, 1):
        try:
            res = convert(pdf, outroot, a.dpi)
            slug = res["slug"]
            while slug in used:
                slug = res["slug"] + "-2"
                res["outdir"] = outroot / slug
                res["outdir"].mkdir(parents=True, exist_ok=True)
            used.add(slug)
            res["slug"] = slug
            outp = write_page(res, outroot, pdf.name)
            nfig = len(list((res["outdir"] / "assets").glob("*.png")))
            rows.append(dict(n=n, slug=slug, title=res["fm"]["title"]
                             if res["fm"] else pdf.stem,
                             npages=res["npages"], nfig=nfig, src=pdf.name))
            print(f"[{n:02d}] {pdf.name} -> {outp} "
                  f"({res['npages']}p, {nfig} figs)", flush=True)
        except Exception as e:
            import traceback
            print(f"[{n:02d}] FAIL {pdf.name}: {e}", flush=True)
            traceback.print_exc()
    write_index(rows, outroot)
    print(f"index -> {outroot/'index.html'}  ({len(rows)} papers)")


if __name__ == "__main__":
    main()
