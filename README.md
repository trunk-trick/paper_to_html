# pdf2reading

A PDF paper in, a single dark HTML page out — typeset for reading rather than
for scrolling a photo of paper.

The download is two halves and both are needed:

- **`adapter/`** — a 66 MB LoRA on `Qwen/Qwen3-4B-Instruct-2507`. It reads one
  page of extracted lines and returns its reading structure: which lines form
  a heading, which form a paragraph, which caption belongs to which figure,
  what is a running head to be dropped, and in what order the blocks read.
- **`corpus/` + `reading_build/`** — the converter that turns a PDF into those
  lines, and the renderer that draws the answer. The renderer is fixed: the
  model cannot change the styling, only the structure. Figures are cropped
  from the PDF itself at 200 dpi, so nothing is redrawn or hallucinated.

A worked example, produced by exactly this checkout, is in
[`examples/How_to_poop/`](examples/How_to_poop/index.html) — open the HTML
next to the original PDF.

## Quick start

```bash
git clone https://github.com/trunk-trick/paper_to_html.git
cd paper_to_html
./setup.sh                                  # venv, torch, deps, base model (8 GB download)
python pdf2reading.py paper.pdf -o out/paper --load-8bit
```

That is the whole install: `setup.sh` builds `.venv`, installs a `cu128`
torch (needed by RTX 50, fine on every older card), installs the rest of
`requirements.txt`, and fetches the base model. Open `out/paper/index.html`.

**On an 8 GB card** (RTX 5060, 4060, 3060) use `--load-8bit`, and install the
quantised path once:

```bash
. .venv/bin/activate && pip install bitsandbytes>=0.46
```

A 16 GB card can drop both the flag and bitsandbytes. A 6 GB card needs
`--load-4bit`; see [Hardware](#hardware) for what that costs.

Older driver: `./setup.sh cu124`. Python 3.10+ and an NVIDIA card are
required. `download_model.py --mirror` or `--modelscope` fetches through a
mirror if the Hugging Face hub is slow.

## If the install fails

**`Missing dependencies for SOCKS support`** — your shell exports a
`socks://` proxy, and pip inside a fresh venv has no PySocks to speak it.
`setup.sh` handles this itself: it uses any `http://` proxy in the
environment, then tries the socks port over HTTP, then tries a direct
connection, and only stops if none of those reach PyPI. In that last case,
give it the HTTP port of your proxy tool:

```bash
HTTPS_PROXY=http://127.0.0.1:7890 ./setup.sh   # Clash 7890, v2rayN 10809
P2H_NO_PROXY=1 ./setup.sh                      # if no proxy is needed
```

**torch: no matching distribution** — check the Python version; wheels exist
for 3.10–3.14 (measured: `torch 2.11.0+cu128` on Python 3.14). An older
Python: `PYTHON=python3.12 ./setup.sh`.

**torch download is slow** — point it at a mirror:
`TORCH_INDEX=https://mirror.example/pytorch/whl/cu128 ./setup.sh`.

## Use

```bash
python pdf2reading.py paper.pdf -o out/paper
# out/paper/index.html, out/paper/img/*.png, plus rec.json and labels.json

python pdf2reading.py paper.pdf -o out/paper --load-8bit    # 8 GB card
python pdf2reading.py paper.pdf -o out/paper --load-4bit    # 6 GB card
python pdf2reading.py paper.pdf -o out/paper --gold         # no model
```

`--gold` renders the *converter's own* structure. It is the self-test: if a
page looks wrong in `--gold` too, the bug is in the renderer, not the model.

As a service:

```bash
python serve.py --port 8000                 # add --load-8bit on an 8 GB card
curl -F file=@paper.pdf http://localhost:8000/convert      # -> {"id": ...}
curl http://localhost:8000/jobs/<id>                        # -> {"state": "done", "page": ...}
```

One job at a time: a conversion owns the GPU while it runs.

## Hardware

Measured on an RTX 4090 with a 78-line page, 1,024-token answer:

| card | flag | base model | peak VRAM | seconds/page |
|---|---|---|---|---|
| 16 GB+ (4060 Ti 16G, 5060 Ti 16G, 4090) | *(none)* | bf16 8.2 GB | 8.9 GiB | 38 |
| 8 GB (5060, 4060, 3070) | `--load-8bit` | int8 4.3 GB | 5.5 GiB | 95 |
| 6 GB | `--load-4bit` | nf4 2.4 GB | 3.9 GiB | 42 |

**Prefer `--load-8bit` on an 8 GB card.** It produced byte-identical structure
to the bf16 run on the sample paper, for about 2.5x the time. 4-bit is the
fallback when even that does not fit, and it *does* change the answer: on the
same sample it merged five blocks and lost a caption, so the figure vanished.

Peak VRAM moves with page length, because the KV cache grows with the page.
The numbers above are a median page; a 300-line page roughly doubles them.

An RTX 50 card needs a `cu128` or newer torch wheel -- `torch 2.7` is the
first build with sm_120 kernels. `setup.sh` defaults to it.

A 300-page book is an overnight job, not an interactive one.

## What the model actually decides

The converter first extracts every text line with its box, font size and
weight, in a naive top-to-bottom sweep — the way a scanner would see them,
which interleaves the two columns of a typical paper. That page becomes the
model's input. The answer is a list of blocks:

```json
[{"r": "h1", "l": [10]}, {"r": "p", "l": [11, 12, 13, 14, 15]},
 {"r": "cap", "l": [34]}, {"r": "drop", "l": [0, 1]}]
```

Roles: `h1`, `h2`, `p`, `cap`, `kw`, `small`, `drop`. Every line index
appears exactly once; a line the model leaves out is repaired to `drop`, which
is visible as a missing line rather than a silently merged paragraph.

Because the answer is indices and not text, the model cannot mistype, reword,
or invent a sentence. Every character in the output comes from the PDF.

**Figures are not the model's job.** The converter already computes each
caption's figure region while it looks for captions, and that region is
stored with the caption; the renderer falls back to the PDF content-stream
order (drawings painted before the caption's own text), then to geometry.
Measured over 287 captions in 17 papers: 82% get their box from the
converter, 1.7% end with no artwork at all, and 6.8% of the converter-sourced
boxes contain body text.

## Measured quality

Held-out pages from the same distribution as training (150 pages, one batch
used for both rows, so the comparison is paired):

| | run1 | **run2 (shipped)** |
|---|---|---|
| output parses as JSON | 98.0% | **98.7%** |
| every line covered | 74.7% | **87.3%** |
| role accuracy | 79.1% | **82.8%** |
| **whole page structure exact** | 12.7% | **24.0%** |
| adjacency F1 | 82.7% | **86.2%** |

"Whole page exact" is all-or-nothing per page: one wrong block scores zero,
which is why it reads low even though 83% of individual lines carry the right
role. Expect the render to be usable on most pages and wrong at some
specific place on many.

## Known limitations

- **A wrapped section heading is often split.** `2 How Should the Buttocks Be
  Aligned with the Pan So As Not to Defecate Outside It?` across two lines
  can come back as `h1` + `h2`. Same for numbered headings whose number sits
  on its own line.
- **Thin subsections may merge into the paragraph below.** `2.5.1` and
  `2.5.2` are sometimes absorbed, which costs a table-of-contents entry.
- **Dense appendix pages are the weak spot.** A page that serialises past the
  context first loses its artwork rectangles -- that alone takes a measured
  appendix page from 73,489 to 15,226 tokens -- and is then split at the
  midpoint, so a block cannot span the split. A page the model answers by
  dropping most of its lines is split too, but only when it drops clearly more
  than the converter itself does on the same page.
- **Non-English text:** trained on English papers. It will run, but the
  heuristics behind caption and heading detection are English-shaped.
- The renderer is academic-paper shaped: title block, abstract, sections,
  captions, references. It is not a general layout engine.

## Files

```
adapter/            LoRA weights (bf16) + tokenizer, 66 MB
corpus/emit.py      PDF -> page records (lines, boxes, captions, figure boxes)
corpus/to_sft.py    page record -> the exact prompt format the model was trained on
corpus/infer.py     the pipeline: extract, generate, repair, render
reading_build/build.py     the rule converter underneath emit.py
reading_build/render.py    blocks + PDF -> HTML
reading_build/theme.css    every visual decision
reading_build/TASTE.md     why those decisions, in prose
reading_build/vocab.txt    word counts, used to undo end-of-line hyphenation
pdf2reading.py      CLI wrapper
serve.py            HTTP wrapper
setup.sh            environment
examples/           a paper converted by this exact checkout
```

## Training

- Base `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), LoRA r=16, alpha=32,
  dropout 0.05 on 7 projection modules across 36 layers: 33.0 M trainable
  parameters (0.81%).
- Data: 16,436 pages, one epoch, max length 12,288, 2,251 steps, ~9.5 h on
  one RTX 4090. Train loss 0.047, val loss 0.021.
- Labels come from the rule converter's own paragraph decisions, split back
  to typeset lines — plus synthetic pages generated to cover layouts the
  corpus lacked, so the model sees structure rather than only the
  converter's favourite shapes.
- `vocab.txt` was rebuilt over the training pages: 77,488 words.
