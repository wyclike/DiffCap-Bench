#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ==============================
# Prompt
# ==============================

PROMPT_YAML = "./prompt/inference.yaml"

def load_prompt_template(path: str = PROMPT_YAML) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["template"]

MAX_PIXELS = 1024 * 1024
MIN_PIXELS = 336 * 336

# ==============================
# Image Utilities
# ==============================

def load_and_encode(source: str):
    """
    Load an image from a local path or HTTP(S) URL, resize so that the total
    pixel count stays within [MIN_PIXELS, MAX_PIXELS] while preserving aspect
    ratio, and return (base64_jpeg_str, meta_dict).
    Returns (None, None) on any failure.
    """
    try:
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, timeout=20)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")

        w, h = img.size
        pixels = w * h

        if pixels > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / pixels)
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((w, h), Image.LANCZOS)
        elif pixels < MIN_PIXELS:
            scale = math.sqrt(MIN_PIXELS / pixels)
            w, h = int(w * scale), int(h * scale)
            img = img.resize((w, h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        g = math.gcd(w, h)
        meta = {"width": w, "height": h, "ratio": f"{w // g}:{h // g}"}
        return b64, meta

    except Exception as exc:
        print(f"[load_and_encode] Failed ({source}): {exc}")
        return None, None


# ==============================
# API Call
# ==============================

def call_model(
    client: OpenAI,
    model: str,
    src: str,
    tgt: str,
    prompt_template: str,
    max_retry: int = 3,
) -> str | None:
    b64_src, meta_src = load_and_encode(src)
    b64_tgt, meta_tgt = load_and_encode(tgt)

    if meta_src is None or meta_tgt is None:
        return None

    prompt = prompt_template.format(meta_src["ratio"], meta_tgt["ratio"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_src}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_tgt}"}},
            ],
        }
    ]

    for attempt in range(1, max_retry + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            return completion.choices[0].message.content
        except Exception as exc:
            print(f"[call_model] Attempt {attempt}/{max_retry} failed: {exc}")
            if attempt < max_retry:
                time.sleep(2 ** attempt)

    return None


# ==============================
# CSV Helpers
# ==============================

def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(data: list[dict], path: str) -> None:
    if not data:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


# ==============================
# Row Worker
# ==============================

def process_row(idx: int, row: dict, client: OpenAI, model: str, prompt_template: str):
    src = row.get("src_url") or row.get("local_path_1")
    tgt = row.get("tgt_url") or row.get("local_path_2")

    if not src or not tgt:
        print(f"[{idx}] Skipping: missing image path")
        return None, idx

    try:
        result = call_model(client, model, src, tgt, prompt_template)
        status = "done" if result is not None else "empty"
        print(f"[{idx}] {status}")
        return result, idx
    except Exception as exc:
        print(f"[{idx}] Error: {exc}")
        return None, idx


# ==============================
# Main
# ==============================

def main(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt_template = load_prompt_template(args.prompt_yaml)
    print(f"Loaded prompt from {args.prompt_yaml}")

    data = read_csv(args.csv_file)
    if not data:
        print("No data found in CSV.")
        return

    print(f"Loaded {len(data)} rows from {args.csv_file}")

    end = args.end_point if args.end_point != -1 else len(data)
    subset = data[args.start_point:end]
    for row in subset:
        row[args.col] = ""

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_row, i, row, client, args.model, prompt_template): i
            for i, row in enumerate(subset)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                result, _ = future.result()
                if result is not None:
                    subset[i][args.col] = result
            except Exception as exc:
                print(f"[{i}] Unexpected error: {exc}")

    os.makedirs(args.result_dir, exist_ok=True)
    out_path = os.path.join(args.result_dir, f"result_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    write_csv(subset, out_path)
    print(f"Saved to {out_path}")


# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare image pairs from a CSV and generate change descriptions."
    )
    parser.add_argument("--prompt_yaml",  type=str,  default="./prompt/inference.yaml",
                        help="Path to the prompt YAML file.")
    parser.add_argument("--csv_file",     type=str,  default="./Data/test.csv")
    parser.add_argument("--result_dir",   type=str,  required=True)
    parser.add_argument("--model",        type=str,  default="gpt-4o")
    parser.add_argument("--col",          type=str,  default="inference_result",
                        help="Column name to write results into.")
    parser.add_argument("--start_point",  type=int,  default=0)
    parser.add_argument("--end_point",    type=int,  default=-1)
    parser.add_argument("--max_workers",  type=int,  default=8)
    parser.add_argument("--api_key",      type=str,  default=None,
                        help="API key (default: OPENAI_API_KEY env var).")
    parser.add_argument("--base_url",     type=str,  default=None,
                        help="API base URL (default: OPENAI_BASE_URL env var).")
    main(parser.parse_args())
