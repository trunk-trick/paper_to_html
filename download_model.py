#!/usr/bin/env python3
"""Fetch the base model the adapter was trained on.

The adapter stores only the LoRA weights (66 MB); Qwen3-4B itself is about
8 GB and comes from the hub, or from ./models if you already have it.

    python download_model.py                 # Hugging Face
    python download_model.py --mirror        # hf-mirror.com (mainland China)
    python download_model.py --modelscope    # ModelScope mirror

The files go into the Hugging Face cache, which is where `from_pretrained`
looks for them, so afterwards the converter runs offline:

    HF_HUB_OFFLINE=1 python pdf2reading.py paper.pdf -o out/paper
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--mirror", action="store_true",
                    help="download through hf-mirror.com")
    ap.add_argument("--modelscope", action="store_true",
                    help="download through ModelScope")
    a = ap.parse_args()

    if a.modelscope:
        from modelscope import snapshot_download
        p = snapshot_download(a.model)
        print("downloaded to", p)
        if not os.environ.get("HF_HUB_OFFLINE"):
            print("note: ModelScope keeps its own cache; point HF at it with")
            print(f"  export HF_HOME={os.path.dirname(p)}")
        return

    if a.mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download
    p = snapshot_download(a.model)
    print("downloaded to", p)


if __name__ == "__main__":
    main()
