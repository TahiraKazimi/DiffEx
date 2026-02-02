# DiffEx
**DiffEx (CVPR 2025) – Official Repository**

This repository contains the code for DiffEx, a diffusion-based framework for explaining image classifiers using semantic image edits.


## Pipeline Overview


## Minimal Dependency Setup:
```bash
pip install torch torchvision transformers diffusers pillow
```

---
## Run the scripts **in order**:

#### 1. Extract Semantic Categories

Use this only if you want to automatically extract high-level semantic categories from images.

```bash
python parent_semantic_ext.py
```
#### 2. Expand Categories into Nested Taxonomy

Converts flat categories from first step output into nested, fine-grained semantic keywords.
```bash
python expand_categories.py \
  --in_json data_parent_semantic.json \
  --out_json data_nested.json
```

Output: nested taxonomy JSON (required by DiffEx).

#### 3. Run DiffEx

Main script. Applies diffusion-based edits and measures classifier sensitivity.
```bash
python diffex_run.py \
  --taxonomy_json data_nested.json \
  --image_path 0004.png \
  --out_json diffex_output.json \
  --clip_labels "male" "female" \
  --clip_target_index 1
```


