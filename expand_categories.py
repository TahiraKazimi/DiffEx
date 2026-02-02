#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI


SYSTEM_PROMPT = """You expand keywords into nested, fine-grained, prompt-friendly sub-keywords for diffusion-based image editing.

Return ONLY valid JSON. No markdown, no explanations.

You will be given:
- domain: string
- category: string
- items: list of strings

You must return a JSON object mapping each item -> list of expanded sub-keywords:

{
  "<item1>": ["sub1", "sub2", ...],
  "<item2>": ["sub1", "sub2", ...],
  ...
}

Rules:
- Use short noun phrases / tags suitable for prompt-based editing.
- Keep expansions visually grounded.
- Avoid duplicates.
- If an item is hard to expand, return a reasonable set anyway (or [] as last resort).
- Do NOT include any extra keys besides the given items.
"""


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    obj = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if obj:
        return json.loads(obj.group(1))

    raise ValueError("Could not parse JSON from model output.")


def normalize_list(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        if not isinstance(x, str):
            continue
        s = " ".join(x.strip().split()).lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--items_per_call", type=int, default=6, help="How many items to expand per API call")
    ap.add_argument("--max_subkeywords", type=int, default=25, help="Target max sub-keywords per item")
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        inp = json.load(f)

    domain = inp.get("domain")
    categories = inp.get("categories")

    if not isinstance(domain, str) or not isinstance(categories, dict):
        raise SystemExit("Input must have keys: domain (str) and categories (object).")

    client = OpenAI()  # uses OPENAI_API_KEY env var

    out_categories: Dict[str, Dict[str, List[str]]] = {}

    for category, items in categories.items():
        if not isinstance(category, str) or not isinstance(items, list):
            continue
        items = [x for x in items if isinstance(x, str) and x.strip()]
        if not items:
            out_categories[category] = {}
            continue

        print(f"[info] Category '{category}' with {len(items)} items")

        merged_for_cat: Dict[str, List[str]] = {it: [] for it in items}

        for chunk_idx, item_chunk in enumerate(chunk_list(items, args.items_per_call), start=1):
            print(f"  [info]  chunk {chunk_idx}: {item_chunk}")

            payload = {
                "domain": domain,
                "category": category,
                "max_subkeywords_per_item": args.max_subkeywords,
                "items": item_chunk,
            }

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ]

            kwargs = {
                "model": args.model,
                "messages": messages,
            }

            # gpt-5-nano does NOT support temperature overrides
            if not args.model.startswith("gpt-5"):
                kwargs["temperature"] = args.temperature

            resp = client.chat.completions.create(**kwargs)

            raw = resp.choices[0].message.content or ""
            parsed = extract_json(raw)

            if not isinstance(parsed, dict):
                raise SystemExit(f"Model returned non-object JSON for category '{category}'.")

            # Merge results for this chunk
            for it in item_chunk:
                vals = parsed.get(it, [])
                if isinstance(vals, list):
                    merged_for_cat[it] = normalize_list(vals)[: args.max_subkeywords]
                else:
                    merged_for_cat[it] = []

        out_categories[category] = merged_for_cat

    out_obj = {"domain": domain, "categories": out_categories}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"[ok] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
