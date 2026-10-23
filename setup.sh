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
proxy_of() { eval "printf '%s' \"\${$1:-}\""; }

# Does a HEAD request through this HTTP proxy succeed?  Proves TCP, CONNECT,
# TLS and an HTTP status in well under a second, and fails just as fast when
# the url is not an HTTP proxy at all.
probe_proxy() {
    command -v curl >/dev/null 2>&1 || return 1
    curl -s --max-time 8 --proxy "$1" -f -I -o /dev/null https://pypi.org/simple/
}

# The same test with no proxy: TCP + TLS handshake only, no download, so the
# answer does not depend on the index page's size.  IPv4 first, because on a
# network with a broken IPv6 route getaddrinfo's default order hangs for tens
# of seconds before falling back.  The alarm bounds DNS, which socket timeouts
# do not cover.
probe_direct() {
    "$PY" - <<'EOF' 2>/dev/null
import signal, socket, ssl
signal.alarm(15)
ctx = ssl.create_default_context()
for fam in (socket.AF_INET, socket.AF_INET6):
    try:
        s = socket.socket(fam, socket.SOCK_STREAM)
        s.settimeout(4)
        s.connect(socket.getaddrinfo("pypi.org", 443, fam, socket.SOCK_STREAM)[0][4])
        ctx.wrap_socket(s, server_hostname="pypi.org").close()
        break
    except Exception:
        continue
else:
    raise SystemExit(1)
EOF
}

socks=""
for v in ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy; do
    case "$(proxy_of "$v")" in *socks*) socks="$socks $v=$(proxy_of "$v")" ;; esac
done
if [ -n "$socks" ]; then
    # Any http(s):// proxy, not just the first variable that is set: a shell
    # can export HTTPS_PROXY=socks:// and HTTP_PROXY=http:// on the same host.
    http=""
    for v in HTTPS_PROXY https_proxy HTTP_PROXY http_proxy; do
        val="$(proxy_of "$v")"
        case "$val" in "" | *socks*) ;; *) http="$val"; break ;; esac
    done
    first="$(echo "$socks" | sed -n '1s/^ *[A-Za-z_]*=//p')"
    same="$(printf '%s' "$first" | sed -E 's#^socks5?h?://#http://#')"
    if [ "${P2H_NO_PROXY:-0}" = 1 ]; then
        echo "note: P2H_NO_PROXY=1; pip will run with no proxy"
        unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
    elif [ -n "$http" ] && probe_proxy "$http"; then
        echo "note: socks proxy present; using the HTTP proxy $http for pip"
        unset ALL_PROXY all_proxy
        # Overwrite every scheme-specific variable: pip reads HTTPS_PROXY for
        # https urls, and that one may still hold the socks url.
        export HTTPS_PROXY="$http" https_proxy="$http" HTTP_PROXY="$http" http_proxy="$http"
    elif [ "$same" != "$first" ] && probe_proxy "$same"; then
        echo "note: the socks proxy also answers HTTP on the same port;"
        echo "      using $same for pip"
        unset ALL_PROXY all_proxy
        export HTTPS_PROXY="$same" HTTP_PROXY="$same" https_proxy="$same" http_proxy="$same"
    elif probe_direct; then
        echo "note: socks proxy present, but PyPI is reachable directly;"
        echo "      ignoring the proxy for this install.  The first download"
        echo "      may pause for a minute while pip times out IPv6 addresses."
        unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
    else
        cat >&2 <<EOF
error: a socks:// proxy is set and a fresh venv has no PySocks yet:
         $(echo "$socks" | sed 's/^ //')
       pip stops before downloading, because it cannot speak socks://.
       This script tried the HTTP proxies and the direct connection and
       both failed, so pip has no route it can use.

       What to do: start (or open) your proxy tool and point pip at its
       HTTP port -- Clash 7890, v2rayN 10809, sing-box 7890 are usual:

              HTTPS_PROXY=http://127.0.0.1:7890 ./setup.sh

       If you know PyPI works without any proxy, force that:

              P2H_NO_PROXY=1 ./setup.sh

       Proxies currently set:
EOF
        for v in ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy; do
            val="$(proxy_of "$v")"
            [ -n "$val" ] && echo "           $v=$val" >&2
        done
        exit 1
    fi
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
