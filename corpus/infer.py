#!/usr/bin/env python3
"""A PDF in, a reading-edition HTML out — the model in the middle.

The model never sees the converter's answer: the page is serialised to lines
with `to_sft.example`, which is the exact function that built the training
inputs, so train and inference cannot drift apart.  What comes back is an
array of blocks per page; it is repaired (see `parse_labels`) and handed to
`render.py`, which is the whole of the styling.

    python3 infer.py --pdf X.pdf --adapter run1/final --out out/X
    python3 infer.py --pdf X.pdf --adapter run1/final --out out/X --load-4bit
    python3 infer.py --pdf X.pdf --gold --out out/X      # teacher, no model

`--load-4bit` halves the base model's footprint (about 2.4 GB instead of 8.2),
which is what makes an 8 GB card enough; the answer is unaffected in the
measurements so far, since the adapter already runs in bf16.

`--gold` is the self-test: it renders the converter's own labels through the
same path, so a bad page in gold mode is a renderer bug, not a model bug.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# the sibling checkout of the renderer, wherever this tree lives
sys.path.insert(0, str(HERE.parent / "reading_build"))

import emit as emitmod            # noqa: E402
import to_sft                     # noqa: E402

ROLES = {"h1", "h2", "p", "cap", "kw", "small", "drop"}


def parse_labels(text, nlines):
    """Model text -> blocks covering every line exactly once.

    A generated array is not trustworthy in the small: it can repeat an
    index, invent one past the end, or stop early (the model copied the
    "one index exactly once" rule, but copying is not obeying).  Repair is
    deterministic and biased to safety — an unclaimed line becomes a `drop`
    block, never silently merged into a neighbour, because a merged line is
    a wrong paragraph and a dropped line is a visible loss.
    """
    m = re.search(r"\[", text)
    if not m:
        return None, nlines
    # raw_decode reads exactly one value and ignores whatever follows.  A
    # greedy /\[.*\]/ instead swallowed any bracket the model wrote after its
    # answer -- with a generous token budget it sometimes keeps talking -- and
    # turned a good answer into "unparseable".
    body = text[m.start():]
    try:
        arr, _ = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError:
        # a truncated array: close it at the last complete object
        cut = body.rfind("}")
        try:
            arr = json.loads(body[:cut + 1] + "]") if cut > 0 else []
        except json.JSONDecodeError:
            return None, nlines
    if not isinstance(arr, list):
        return None, nlines

    seen, blocks = set(), []
    for b in arr:
        if not isinstance(b, dict):
            continue
        idx = []
        for i in b.get("l") or []:
            if isinstance(i, int) and 0 <= i < nlines and i not in seen:
                idx.append(i)
                seen.add(i)
        if not idx:
            continue
        r = b.get("r")
        blocks.append({"r": r if r in ROLES else "p", "l": idx})
    missing = [i for i in range(nlines) if i not in seen]
    blocks.extend({"r": "drop", "l": [i]} for i in missing)
    return blocks, len(missing)


def load_model(adapter, load_4bit=False, load_8bit=False):
    """Load the base model and put the adapter on it.

    The adapter was trained against a bf16 base.  Quantising the base to 4
    bits changes the answer -- on a two-page sample it merged five blocks and
    lost the caption, so the figure disappeared -- which makes 4-bit a
    last-resort path for an 8 GB card, not a free one.  8-bit is closer.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter)
    base = tok.init_kwargs.get("name_or_path") or "Qwen/Qwen3-4B-Instruct-2507"
    kw = {"device_map": "cuda"}
    if load_4bit or load_8bit:
        from transformers import BitsAndBytesConfig
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def generate(tok, model, prompts, max_new_tokens):
    """Generate one answer per prompt, never batching a long page with peers.

    The page is the unit of work and pages differ hugely in length: a poster
    or an appendix page can be ten times the median.  Batching four of them
    pads every row to the longest, so one outlier page multiplies the KV
    cache of the whole batch and the run dies with an OOM -- taking the rest
    of the paper with it.  Group by length instead, keep the padded token
    budget per step bounded, and treat a single oversized page as a local
    miss rather than a paper-level failure.
    """
    import torch
    texts = [tok.apply_chat_template(p, tokenize=False,
                                     add_generation_prompt=True)
             for p in prompts]
    enc = [tok(t)["input_ids"] for t in texts]
    MAX_PAGE = 32000       # beyond this a page cannot be served on this card
    BUDGET = 9000          # padded tokens per generation step
    outs = [""] * len(texts)
    groups, cur, curmax = [], [], 0
    for idx in sorted(range(len(texts)), key=lambda i: len(enc[i])):
        L = len(enc[idx])
        if L > MAX_PAGE:
            print(f"skip page: {L} tokens > {MAX_PAGE}", flush=True)
            continue
        newmax = max(curmax, L)
        if cur and (len(cur) >= 4 or newmax * (len(cur) + 1) > BUDGET):
            groups.append(cur)
            cur, curmax, newmax = [], 0, L
        cur.append(idx)
        curmax = newmax
    if cur:
        groups.append(cur)

    for g in groups:
        inputs = tok([texts[i] for i in g], return_tensors="pt",
                     padding=True, padding_side="left").to(model.device)
        n = inputs["input_ids"].shape[1]
        try:
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j, i in enumerate(g):
                outs[i] = tok.decode(gen[j][n:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            print(f"OOM on batch {g}, retrying one page at a time", flush=True)
            for i in g:
                one = tok([texts[i]], return_tensors="pt",
                          padding=True, padding_side="left").to(model.device)
                m = one["input_ids"].shape[1]
                try:
                    with torch.no_grad():
                        o = model.generate(**one,
                                           max_new_tokens=min(max_new_tokens,
                                                              2048),
                                           do_sample=False,
                                           pad_token_id=tok.pad_token_id)
                    outs[i] = tok.decode(o[0][m:], skip_special_tokens=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"skip page {i}: {m} tokens OOM even alone",
                          flush=True)
                finally:
                    del one
                    torch.cuda.empty_cache()
    return outs


CTX = 32768        # Qwen3-4B's context, prompt plus answer
MAX_PAGE = 30000   # a page larger than this has to be split


def prompt_for(rec, pno, lo, hi, with_graphics=True):
    """The model's prompt for lines [lo, hi) of one page.

    A sliced page keeps only the artwork that lies in that band.  One page in
    the sample set carries 2,083 rectangles -- a table drawn rule by rule --
    which costs more tokens than its 397 lines; copying them into every slice
    would keep every slice over the context limit and the split would never
    terminate.
    """
    pg = rec["pages"][pno]
    if lo == 0 and hi >= len(pg["lines"]) and with_graphics:
        ex = to_sft.example(rec, pno)
    else:
        lines = pg["lines"][lo:hi]
        ys = [l["bbox"][1] for l in lines] + [l["bbox"][3] for l in lines]
        y0, y1 = (min(ys) - 24, max(ys) + 24) if ys else (-1e9, 1e9)
        gs = [g for g in pg["graphics"] if g[3] >= y0 and g[1] <= y1] \
            if with_graphics else []
        sub = {"pages": [dict(pg, lines=lines, graphics=gs)]}
        ex = to_sft.example(sub, 0)
    return [m for m in ex["messages"] if m["role"] != "assistant"]


def shift(blocks, off):
    return [{"r": b["r"], "l": [i + off for i in b["l"]]} for b in blocks]


def labels_span(tok, model, rec, pno, lo, hi, max_new, depth=0):
    """Labels for lines [lo, hi), splitting a page that cannot fit the context.

    A dense page serialises past the model's context -- one appendix page in
    the sample set is 73k tokens -- and the old code skipped it, printing the
    page as blank.  Split at the midpoint instead: a block cannot span the
    split, which is a real but small loss against losing the whole page.
    Indices come back absolute within the page.
    """
    n = hi - lo
    if n <= 0:
        return []

    def ntok(pr):
        t = tok.apply_chat_template(pr, tokenize=False,
                                    add_generation_prompt=True)
        return len(tok(t)["input_ids"])

    pr = prompt_for(rec, pno, lo, hi)
    L = ntok(pr)
    if L + 128 > CTX:
        # artwork is a hint; the lines are the job
        pr2 = prompt_for(rec, pno, lo, hi, with_graphics=False)
        L2 = ntok(pr2)
        if L2 < L:
            pr, L = pr2, L2
    if L + max_new + 64 > CTX and n > 8 and depth < 6:
        mid = lo + n // 2
        return (labels_span(tok, model, rec, pno, lo, mid, max_new, depth + 1)
                + labels_span(tok, model, rec, pno, mid, hi, max_new,
                              depth + 1))
    budget = min(max_new, max(256, CTX - L - 64))
    out = generate(tok, model, [pr], budget)[0]
    lab, _ = parse_labels(out, n)
    if lab is None:
        # Truncation, a stray bracket, a model that starts over: an
        # unparseable answer on a long span is better halved than dropped.
        if n > 8 and depth < 6:
            mid = lo + n // 2
            print(f"split span {lo}:{hi} after an unparseable answer",
                  flush=True)
            return (labels_span(tok, model, rec, pno, lo, mid, max_new,
                                depth + 1)
                    + labels_span(tok, model, rec, pno, mid, hi, max_new,
                                  depth + 1))
        return [{"r": "drop", "l": list(range(lo, hi))}]
    lab = shift(lab, lo)
    # A page can also come back as "all of this is a drop block", which parses
    # perfectly and is still useless.  But plenty of pages are mostly dropped
    # on purpose -- a table's cell fragments, a figure's axis labels -- so
    # compare with the converter's own structure for this span: only an answer
    # that drops much more than the converter does is a failure worth
    # splitting.  One split per page (depth 0): a page that stays lazy in
    # halves is usually dense text the model genuinely cannot structure, and
    # each extra split costs another full generation.
    d_lab = dropped(lab)
    if n > 8 and depth == 0 and d_lab * 2 > n:
        d_gold = gold_drops(rec, pno, lo, hi)
        if d_lab > d_gold + n // 4:
            mid = lo + n // 2
            print(f"split span {lo}:{hi}: model dropped {d_lab}, "
                  f"converter {d_gold} of {n}", flush=True)
            alt = (labels_span(tok, model, rec, pno, lo, mid, max_new,
                               depth + 1)
                   + labels_span(tok, model, rec, pno, mid, hi, max_new,
                                 depth + 1))
            if dropped(alt) < d_lab:
                return alt
            print(f"  kept the whole-span answer ({d_lab} dropped)",
                  flush=True)
    return lab


def dropped(blocks):
    """Line indices the answer threw away -- the renderer cannot show them."""
    return sum(len(b["l"]) for b in blocks if b["r"] == "drop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--gold", action="store_true",
                    help="use the converter's labels instead of the model")
    ap.add_argument("--dpi", type=int, default=24,
                    help="resolution of the structural pass (not the crops)")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit base model: about 2.4 GB instead of 8.2; "
                         "answers can differ from the bf16 run")
    ap.add_argument("--load-8bit", action="store_true",
                    help="8-bit base model: about 4.3 GB, closer to bf16")
    a = ap.parse_args()
    if not (a.gold or a.adapter):
        ap.error("--adapter PATH or --gold is required")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "rec.json"
    with tempfile.TemporaryDirectory() as tmp:
        rec = emitmod.emit(a.pdf, Path(tmp) / "struct", a.dpi)
    rec_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

    if a.gold:
        labels = [render_gold(pg) for pg in rec["pages"]]
    else:
        tok, model = load_model(a.adapter, a.load_4bit, a.load_8bit)
        prompts = [prompt_for(rec, p, 0, len(rec["pages"][p]["lines"]))
                   for p in range(len(rec["pages"]))]
        texts = generate(tok, model, prompts, a.max_new)
        labels, misses, retry = [], [], []
        for pno, (text, pg) in enumerate(zip(texts, rec["pages"])):
            n = len(pg["lines"])
            lab, miss = parse_labels(text, n) if text else (None, n)
            if lab is None:
                print(f"page {pno}: unparseable output, all lines dropped")
                lab = [{"r": "drop", "l": [i]} for i in range(n)]
                miss = n
            print(f"page {pno}: {n} lines, "
                  f"{len(lab)} blocks, {miss} repaired->drop")
            labels.append(lab)
            # The retry is scored on dropped lines, which is what the reader
            # loses; `miss` only counts the repairs, and is zero for a page
            # the model answered with drop blocks of its own.
            dropped_here = dropped(lab)
            misses.append(dropped_here)
            # A truncated answer and an oversized page both lose lines, and
            # both are fixable: retry with twice the budget, which splits the
            # page if the context still cannot hold it.  A page answered with
            # drop blocks parses cleanly, so compare the drops with the
            # converter's own for that page -- otherwise the worst answers are
            # the ones never retried.
            lazy = (dropped_here * 2 > n
                    and dropped_here > gold_drops(rec, pno) + n // 4)
            if miss > max(3, int(0.03 * n)) or lazy:
                retry.append(pno)
        for pno in retry:
            n = len(rec["pages"][pno]["lines"])
            lab = labels_span(tok, model, rec, pno, 0, n, a.max_new * 2)
            miss = dropped(lab)
            if miss < misses[pno]:
                print(f"page {pno}: retry {misses[pno]} -> {miss} dropped")
                labels[pno] = lab
            else:
                print(f"page {pno}: retry no better ({miss} dropped)")

    lab_path = out / "labels.json"
    lab_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    cmd = [sys.executable, str(HERE.parent / "reading_build" / "render.py"),
           "--pdf", a.pdf, "--rec", str(rec_path),
           "--labels", str(lab_path), "--out", str(out)]
    subprocess.run(cmd, check=True)


def gold_drops(rec, pno, lo=0, hi=None):
    """Lines in [lo, hi) the converter itself would drop.  Free, no model."""
    pg = rec["pages"][pno]
    hi = len(pg["lines"]) if hi is None else hi
    return sum(len([i for i in b["l"] if lo <= i < hi])
               for b in render_gold(pg) if b.get("r") == "drop")


def render_gold(pg):
    """The converter's own grouping, same shape the model must produce."""
    order = {}
    for pos, ln in enumerate(pg["lines"]):
        if ln["para"] >= 0:
            order.setdefault(ln["para"], []).append(pos)
    out = []
    for pi in sorted(order):
        mem = sorted(order[pi], key=lambda i: pg["lines"][i].get("ord", i))
        ln = pg["lines"][mem[0]]
        r = ln["role"]
        if r == "h":
            r = "h1" if ln.get("lvl", 1) <= 1 else "h2"
        out.append({"r": r if r in ROLES else "drop", "l": mem})
    placed = {i for b in out for i in b["l"]}
    out.extend({"r": "drop", "l": [i]} for i in range(len(pg["lines"]))
               if i not in placed)
    return out


if __name__ == "__main__":
    main()
