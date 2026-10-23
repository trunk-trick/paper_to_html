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

# A socks:// proxy in the environment breaks every pip call in a fresh venv:
# pip's HTTP stack has no PySocks, so it dies with "Missing dependencies for
# SOCKS support" before it downloads anything.  The same machine almost always
# also exposes an HTTP proxy port, and that one needs no extra package.
socks=""
for v in ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy; do
    eval "val=\${$v:-}"
    case "$val" in *socks*) socks=1 ;; esac
done
if [ -n "$socks" ]; then
    http="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
    case "$http" in
        "" | *socks*)
            cat >&2 <<'EOF'
error: only a socks:// proxy is set, and this venv has no PySocks.
       Either point the HTTP proxy at the same host, or run:

           ALL_PROXY= ./setup.sh

       (pip needs no proxy at all when the index is reachable directly.)
EOF
            exit 1
            ;;
        *)
            echo "note: socks proxy found; using $http for pip instead"
            unset ALL_PROXY all_proxy
            ;;
    esac
fi

"$PY" -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel

echo "== torch ($CUDA_TAG) =="
pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/$CUDA_TAG}"

echo "== inference deps =="
pip install -r requirements.txt

echo "== base model =="
python download_model.py

cat <<EOF

Ready.
  . .venv/bin/activate
  python pdf2reading.py paper.pdf -o out/paper
  python serve.py                # HTTP service on :8000

An 8 GB card: add --load-8bit (identical answers, ~5.5 GiB).
A 6 GB card: --load-4bit.  A 16 GB card: the default bf16 path.
EOF
