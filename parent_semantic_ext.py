#!/usr/bin/env python3
"""
Extract fine-grained editable semantic attributes from a folder of images
using an OpenAI vision-capable model.

Example:
  python extract_semantics.py \
    --image_dir /path/to/images \
    --out_json data_fashion.json \
    --model gpt-4o-mini \
    --max_images 32
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


SUPPORTED_EXTS = {".png", ".jpeg", ".jpg", ".gif", ".webp"}
MAX_IMAGE_MB_DEFAULT = 20


SYSTEM_PROMPT = """You are an expert at discovering fine-grained, editable visual attributes for diffusion-based image editing.

Given a set of images from one domain, infer a taxonomy of important attributes that a text-based editor should control.
Cover diverse categories (e.g., clothing, accessories, hairstyle, face, skin, expression, environment, lighting, pose).
For each category, list concrete attribute values (fine-grained). Avoid vague items like "nice" or "pretty".

Return ONLY valid JSON with this schema:

{
  "domain": "<short domain name>",
  "categories": {
    "<Category Name>": ["attribute value 1", "attribute value 2", "..."],
    "<Category Name 2>": ["..."]
  }
}

Rules:
- JSON must be strictly valid (double quotes, no trailing commas).
- Use lists ([]) not sets ({}).
- Category keys should be concise and human-readable.
- Attribute values should be short noun phrases appropriate for prompt-based editing.
"""


def encode_image_to_data_url(image_path: str, max_mb: int) -> Optional[str]:
    """Encode an image file to a data URL that can be sent to the API."""
    try:
        file_size = os.path.getsize(image_path)
    except OSError:
        return None

    if file_size > max_mb * 1024 * 1024:
        print(f"[skip] {image_path} (>{max_mb}MB)")
        return None

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        print(f"[skip] {image_path} (unsupported: {ext})")
        return None

    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".gif":
        mime = "image/gif"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def list_images(image_dir: str) -> List[str]:
    files = []
    for name in os.listdir(image_dir):
        path = os.path.join(image_dir, name)
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_EXTS:
                files.append(path)
    files.sort()
    return files


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON directly; if the model wraps JSON in text/code fences,
    extract the first {...} block.
    """
    text = text.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) strip markdown code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # 3) find first JSON object-ish block
    obj = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if obj:
        return json.loads(obj.group(1))

    raise ValueError("Could not find valid JSON in the model response.")


def validate_schema(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object.")
    if "domain" not in obj or "categories" not in obj:
        raise ValueError('JSON must contain keys: "domain" and "categories".')
    if not isinstance(obj["domain"], str) or not obj["domain"].strip():
        raise ValueError('"domain" must be a non-empty string.')
    if not isinstance(obj["categories"], dict):
        raise ValueError('"categories" must be an object mapping category -> list of strings.')

    for k, v in obj["categories"].items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("All category names must be non-empty strings.")
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f'Category "{k}" must map to a list of strings.')


def build_messages(image_data_urls: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": "Analyze these images and produce the JSON taxonomy."}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of input images")
    parser.add_argument("--out_json", type=str, required=True, help="Where to write the JSON output")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--max_images", type=int, default=32, help="Max number of images to send")
    parser.add_argument("--max_image_mb", type=int, default=MAX_IMAGE_MB_DEFAULT, help="Skip images larger than this")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise SystemExit(f"image_dir not found: {args.image_dir}")

    paths = list_images(args.image_dir)[: args.max_images]
    if not paths:
        raise SystemExit("No supported images found in the directory.")

    data_urls = []
    for p in paths:
        url = encode_image_to_data_url(p, max_mb=args.max_image_mb)
        if url:
            data_urls.append(url)

    if not data_urls:
        raise SystemExit("All images were skipped (size/format).")

    client = OpenAI()  # uses OPENAI_API_KEY env var

    messages = build_messages(data_urls)

    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
    )

    raw = resp.choices[0].message.content or ""
    print("\n=== Raw model output ===\n")
    print(raw)

    try:
        obj = extract_json_from_text(raw)
        validate_schema(obj)
    except Exception as e:
        print("\n[error] Failed to parse/validate JSON:", str(e))
        print("[hint] Try lowering temperature or reducing the number of images.")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()


