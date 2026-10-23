#!/usr/bin/env bash
# One-shot environment for the PDF -> reading-edition converter.
#
#   ./setup.sh              # CUDA 12.8 wheels (RTX 50 series, and every older card)
#   ./setup.sh cu124        # older driver, CUDA 12.4 wheels
#
# Everything lands in ./.venv and ./models, so the checkout is the install.
set -euo pipefail
cd "$(dirname "$0")"

CUDA_TAG="${1:-cu128}"
PY="${PYTHON:-python3}"

"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit("Python 3.10+ is required (found %s)" % sys.version.split()[0])
EOF

"$PY" -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel

echo "== torch ($CUDA_TAG) =="
pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

echo "== inference deps =="
pip install -r requirements.txt

echo "== base model =="
python download_model.py

cat <<EOF

Ready.
  . .venv/bin/activate
  python pdf2reading.py paper.pdf -o out/paper
  python serve.py                # HTTP service on :8000

An 8 GB card: add --load-4bit.  A 16 GB card: the default bf16 path fits.
EOF
