#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from edit_model import ledits_edit

# Editor Function (can be replaceable with any other editing model e.g. qwen image, controlNet)
class LeditsEditor:
    def edit(self, image, keyword, seed=0):
        torch.manual_seed(seed)
        return ledits_edit(image, keyword)



# CLIP classifier (can be replaced with off-the-shelf classifiers)

class CLIPClassifier:
    """
    CLIP-based "classifier" that scores an image against a list of label prompts.
    score(image) returns probability of the target label index.
    """

    def __init__(
        self,
        labels: List[str],
        target_index: int = 0,
        model_name: str = "openai/clip-vit-base-patch32",
        prompt_template: str = "a photo of a {}",
        device: str | None = None,
    ):
        if not labels:
            raise ValueError("CLIPClassifier requires at least 1 label.")
        if not (0 <= target_index < len(labels)):
            raise ValueError(f"target_index must be within [0, {len(labels)-1}]")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.labels = labels
        self.target_index = target_index
        self.prompt_template = prompt_template

        # Prebuild prompts
        self.prompts = [prompt_template.format(l) for l in labels]

    @torch.no_grad()
    def score(self, image: Image.Image) -> float:
        inputs = self.processor(
            text=self.prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        out = self.model(**inputs)
        # logits_per_image: [1, num_labels]
        logits = out.logits_per_image[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[self.target_index].item())


# DiffEx data structures

@dataclass
class Candidate:
    path: List[str]              # e.g., ["accessories", "glasses", "sunglasses"]
    keyword: str                 # leaf keyword (last element of path)
    mean_score: float            # E[score(edited)]
    base_score: float            # score(original)
    delta: float                 # mean_score - base_score


# Taxonomy helpers

def load_nested_taxonomy(path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Expected nested format:
    {
      "domain": "people",
      "categories": {
        "accessories": { "glasses": ["sunglasses", ...], ... },
        "age": { "infant": ["newborn", ...], ... }
      }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict) or "domain" not in obj or "categories" not in obj:
        raise ValueError("Nested taxonomy JSON must have keys: 'domain' and 'categories'.")

    domain = obj["domain"]
    cats = obj["categories"]

    if not isinstance(domain, str) or not isinstance(cats, dict):
        raise ValueError("'domain' must be str and 'categories' must be an object.")

    return domain, cats


def iter_level1_items(categories: Dict[str, Any]) -> List[Tuple[List[str], str]]:
    """
    Level-1 items are the keys inside each category dict.
    Example:
      categories["accessories"].keys() -> glasses, hat, ...
    Returns:
      ([category, item], item)
    """
    out = []
    for cat_name, cat_obj in categories.items():
        if isinstance(cat_obj, dict):
            for item in cat_obj.keys():
                if isinstance(item, str):
                    out.append(([cat_name, item], item))
    return out


def iter_level2_keywords(categories: Dict[str, Any], parent_path: List[str]) -> List[Tuple[List[str], str]]:
    """
    For parent_path ["accessories", "glasses"], return:
      (["accessories","glasses","sunglasses"], "sunglasses")
    """
    if len(parent_path) != 2:
        return []
    cat, item = parent_path
    cat_obj = categories.get(cat, None)
    if not isinstance(cat_obj, dict):
        return []
    kw_list = cat_obj.get(item, None)
    if not isinstance(kw_list, list):
        return []
    out = []
    for kw in kw_list:
        if isinstance(kw, str) and kw.strip():
            out.append((parent_path + [kw], kw))
    return out


# Scoring

def estimate_candidate_delta(
    image: Image.Image,
    keyword: str,
    editor: LeditsEditor,
    classifier: CLIPClassifier,
    n_samples: int,
    seed: int,
) -> Tuple[float, float, float]:
    """
    Returns (mean_score, base_score, delta).
    Multiple samples mimic different edit randomness.
    """
    base_score = classifier.score(image)

    scores: List[float] = []
    for i in range(n_samples):
        edited = editor.edit(image, keyword=keyword, seed=seed + i)
        s = classifier.score(edited)
        scores.append(s)

    mean_score = sum(scores) / max(1, len(scores))
    delta = mean_score - base_score
    return mean_score, base_score, delta


# DiffEx core

def diffex_run(
    image: Image.Image,
    categories: Dict[str, Any],
    editor: LeditsEditor,
    classifier: CLIPClassifier,
    beam_size: int = 8,
    top_k: int = 25,
    n_samples_per_keyword: int = 4,
    seed: int = 0,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    # ---- Level 1: score each item (edit using item string) ----
    level1: List[Candidate] = []
    for path, item in iter_level1_items(categories):
        mean_score, base_score, delta = estimate_candidate_delta(
            image=image,
            keyword=item,
            editor=editor,
            classifier=classifier,
            n_samples=n_samples_per_keyword,
            seed=seed,
        )
        if abs(delta) >= min_delta:
            level1.append(Candidate(path=path, keyword=item, mean_score=mean_score, base_score=base_score, delta=delta))

    level1.sort(key=lambda c: abs(c.delta), reverse=True)
    beam = level1[:beam_size]

    # ---- Level 2: expand within top beam items ----
    leaf_candidates: List[Candidate] = []
    for parent in beam:
        for leaf_path, kw in iter_level2_keywords(categories, parent.path):
            mean_score, base_score, delta = estimate_candidate_delta(
                image=image,
                keyword=kw,
                editor=editor,
                classifier=classifier,
                n_samples=n_samples_per_keyword,
                seed=seed,
            )
            if abs(delta) >= min_delta:
                leaf_candidates.append(Candidate(path=leaf_path, keyword=kw, mean_score=mean_score, base_score=base_score, delta=delta))

    leaf_candidates.sort(key=lambda c: abs(c.delta), reverse=True)
    top_leaf = leaf_candidates[:top_k]

    return {
        "base_score": classifier.score(image),
        "clip_labels": classifier.labels,
        "clip_target_index": classifier.target_index,
        "level1_top": [asdict(c) for c in beam],
        "top_influential": [asdict(c) for c in top_leaf],
        "stats": {
            "num_level1_scored": len(level1),
            "num_leaf_scored": len(leaf_candidates),
            "beam_size": beam_size,
            "top_k": top_k,
            "n_samples_per_keyword": n_samples_per_keyword,
            "min_delta": min_delta,
        },
    }


# CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy_json", required=True, help="Nested taxonomy JSON")
    ap.add_argument("--image_path", required=True, help="Path to a single image")
    ap.add_argument("--out_json", required=True, help="Output JSON with influential features")

    ap.add_argument("--beam_size", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_delta", type=float, default=0.0)

    # CLIP args
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip_prompt_template", default="a photo of a {}")
    ap.add_argument("--clip_target_index", type=int, default=0)
    ap.add_argument("--clip_labels", nargs="+", required=True, help="Labels to score against (space-separated)")

    args = ap.parse_args()

    domain, categories = load_nested_taxonomy(args.taxonomy_json)

    image = Image.open(args.image_path).convert("RGB")

    editor = LeditsEditor()
    classifier = CLIPClassifier(
        labels=args.clip_labels,
        target_index=args.clip_target_index,
        model_name=args.clip_model,
        prompt_template=args.clip_prompt_template,
    )

    results = diffex_run(
        image=image,
        categories=categories,
        editor=editor,
        classifier=classifier,
        beam_size=args.beam_size,
        top_k=args.top_k,
        n_samples_per_keyword=args.samples,
        seed=args.seed,
        min_delta=args.min_delta,
    )

    out_obj = {
        "domain": domain,
        "image": args.image_path,
        "diffex_results": results,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"[ok] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
