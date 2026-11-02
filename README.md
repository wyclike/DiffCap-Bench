# DiffCapBench
## Repository Structure

```
DiffCapBench/
├── Data/
│   └── test.csv                  # Benchmark dataset (image pairs + expert annotations)
├── inference/
│   └── run_inference.py          # Run a VLM on the benchmark dataset
├── judge/
│   └── judge.py                  # Evaluate model predictions against expert annotations
├── prompt/
│   ├── inference.yaml            # Prompt template for the inference stage
│   └── judge.yaml                # Prompt template for the judging stage
└── Result/                       # Directory for storing output CSV files
```

---

## Dataset

`Data/test.csv` contains 1,075 image pairs with the following columns:

| Column | Description |
|---|---|
| `relation_id` | Unique identifier for each image pair |
| `src_url` | URL of the source image (Image A) |
| `tgt_url` | URL of the target image (Image B) |
| `src_aspect_ratio` | Aspect ratio of the source image |
| `tgt_aspect_ratio` | Aspect ratio of the target image |
| `keypoint_list_with_dimension` | Expert-annotated keypoint list describing ground-truth changes |

The `keypoint_list_with_dimension` field contains structured annotations with two parts:
- **Key Change List** — certain and important visual changes from A to B.
- **Uncertain Change List** —  unconfirmed changes.

---

## Installation

```bash
pip install openai Pillow requests pyyaml tqdm
```

---

## Stage 1 — Inference

`inference/run_inference.py` queries any OpenAI-compatible multimodal API to generate change descriptions for each image pair in the dataset.

### Usage

```bash
python inference/run_inference.py \
    --csv_file   ./Data/test.csv \
    --result_dir ./Result \
    --model      gpt5.2 \
    --col        inference_result \
    --max_workers 8 \
    --api_key    <YOUR_API_KEY> \
    --base_url   https://api.openai.com/v1
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv_file` | `./Data/test.csv` | Path to the input CSV |
| `--result_dir` | *(required)* | Directory to save results |
| `--model` | `gpt5.2` | Model name |
| `--prompt_yaml` | `./prompt/inference.yaml` | Prompt template file |
| `--col` | `inference_result` | Column name for model outputs |
| `--start_point` | `0` | Start row index (inclusive) |
| `--end_point` | `-1` | End row index (exclusive; `-1` = all rows) |
| `--max_workers` | `8` | Number of parallel threads |
| `--api_key` | `$OPENAI_API_KEY` | API key |
| `--base_url` | `$OPENAI_BASE_URL` | API base URL |

The output is saved to `<result_dir>/result_<timestamp>.csv`.

---

## Stage 2 — Judging

`judge/judge.py` evaluates model-predicted descriptions against the expert keypoint annotations using a vision-language judge model. Both source and target images are provided to the judge alongside the textual inputs.

The judge performs:
1. **Forward Checking** — for each expert keypoint, determines whether the prediction *Hit and Correct*, *Hit but Incorrect*, or *Not Hit*.
2. **Backward Checking** — for each extra prediction not matched to a keypoint, classifies it as *Extra Correct Description*, *Matches Uncertain Change List*, or *Hallucination*.

### Usage

```bash
python judge/judge.py \
    --csv_file     ./Result/result_<timestamp>.csv \
    --result_dir   ./Result \
    --model        gemini-2.5-pro \
    --pred_col     inference_result \
    --keypoint_col keypoint_list_with_dimension \
    --col          judge_result \
    --max_workers  8 \
    --api_key      <YOUR_API_KEY> \
    --base_url     https://generativelanguage.googleapis.com/v1beta/openai/
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv_file` | `./Data/test.csv` | Path to the input CSV (should contain inference results) |
| `--result_dir` | *(required)* | Directory to save results |
| `--model` | `gemini-2.5-pro` | Judge model name |
| `--prompt_yaml` | `./prompt/judge.yaml` | Prompt template file |
| `--pred_col` | `inference_result` | Column with model-predicted descriptions |
| `--keypoint_col` | `keypoint_list_with_dimension` | Column with expert annotations |
| `--col` | `judge_result` | Column name for judge outputs |
| `--start_point` | `0` | Start row index (inclusive) |
| `--end_point` | `-1` | End row index (exclusive; `-1` = all rows) |
| `--max_workers` | `8` | Number of parallel threads |
| `--api_key` | `$OPENAI_API_KEY` | API key |
| `--base_url` | `$OPENAI_BASE_URL` | API base URL |

The output is saved to `<result_dir>/judge_<timestamp>.csv`. Each row's `judge_result` field contains a structured JSON object:

```json
{
  "Key_Change_List": [
    {
      "Description": "...",
      "Type": "...",
      "Hit_Status": "Hit and Correct / Hit but Incorrect / Not Hit",
      "Reason": "..."
    }
  ],
  "Extra_Description_List": [
    {
      "Description": "...",
      "Status": "Hallucination / Matches Uncertain Change List / Extra Correct Description",
      "Reason": "..."
    }
  ],
  "Hallucination": {
    "Count": 0,
    "Reason": "..."
  }
}
```

---

## Full Pipeline Example

```bash
# Step 1: Run inference
python inference/run_inference.py \
    --csv_file ./Data/test.csv \
    --result_dir ./Result \
    --model gpt-4o

# Step 2: Judge the predictions
python judge/judge.py \
    --csv_file ./Result/result_<timestamp>.csv \
    --result_dir ./Result \
    --model gemini-2.5-pro
```
