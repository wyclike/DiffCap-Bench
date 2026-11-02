#!/usr/bin/env python3
"""Score saved DiffCap-Bench judge outputs without images or API calls.

All metrics are returned on a 0--1 scale (hallucination rate can exceed 1).
The CLI prints percentages. Overall metrics are per-pair averages; category
F1 defaults to pooled item counts (--category-average micro).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORIES = (
    "composition", "background", "attribute", "text", "camera",
    "expression", "spatial", "image_property", "object", "action_pose",
)
METRICS = ("recall", "precision", "f1", "hallucination_rate")
STATUSES = {
    "Hit and Correct": "correct", "命中且正确": "correct",
    "Hit but Incorrect": "incorrect", "命中且错误": "incorrect",
    "Hit and Incorrect": "incorrect", "Not Hit": "missed", "未命中": "missed",
}


def structured(value):
    """Accept JSON, historical Python literals, or a fenced JSON object; never eval."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("incomplete code fence")
        value = "\n".join(lines[1:-1])
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def field(obj, *names):
    for name in names:
        if name in obj:
            return obj[name]
    raise ValueError("missing field: " + "/".join(names))


def labels(item):
    value = field(item, "Type", "类型")
    parts = re.split(r"[,，;；]", value) if isinstance(value, str) else value
    if not isinstance(parts, list):
        raise ValueError("category labels must be a string or list")
    result = set(x.strip() for x in parts if isinstance(x, str) and x.strip())
    if not result or result - set(CATEGORIES):
        raise ValueError("unknown or empty categories: " + repr(value))
    return result


def metric_values(correct, incorrect, n_key, hallucinations=0):
    if n_key <= 0:
        raise ValueError("reference key list must not be empty")
    denom = correct + incorrect + hallucinations
    return dict(
        recall=correct / n_key,
        precision=correct / denom if denom else 0.0,
        f1=2 * correct / (n_key + denom),
        hallucination_rate=hallucinations / n_key,
    )


def score_sample(row, reference, pred_col=None, judge_col="judge_result",
                 hallu_source="summary"):
    """Return metrics and counts, or an explicit invalid record (never a fake zero)."""
    result = dict(status="invalid", errors=[], warnings=[], metrics=None, categories={})
    try:
        if hallu_source not in ("summary", "list", "strict"):
            raise ValueError("unknown hallucination source")
        ref = structured(reference)
        key_items = field(ref, "关键变化列表", "Key_Change_List", "Key Change List")
        if not isinstance(key_items, list) or not key_items:
            raise ValueError("missing or empty reference key list")
        ref_labels = [labels(x) for x in key_items]
        n_key = len(key_items)
        result["counts"] = dict(n_key=n_key)
        column = pred_col
        if column is None:
            column = next((k for k in ("prediction_result", "inference_result") if k in row), None)
        if column is None or column not in row:
            raise ValueError("missing prediction column; use --pred-col")
        prediction = row[column]
        if not isinstance(prediction, str):
            raise ValueError("prediction must be a string (null is missing, not empty)")
        if not prediction.strip():
            counts = Counter(missed=n_key)
            h = 0
            categories = {c: ["missed" for cats in ref_labels if c in cats]
                          for c in CATEGORIES}
            result["warnings"].append("empty_prediction_all_metrics_zero")
        else:
            judge = structured(field(row, judge_col))
            if not isinstance(judge, dict):
                raise ValueError("missing or non-object judge result")
            items = field(judge, "Key_Change_List", "关键变化列表", "Key Change List")
            if not isinstance(items, list) or len(items) != n_key:
                raise ValueError("judge/reference item count mismatch")
            counts = Counter()
            categories = {c: [] for c in CATEGORIES}
            for i, (item, reference_item, cats) in enumerate(zip(items, key_items, ref_labels)):
                raw_status = field(item, "Hit_Status", "命中状态")
                if raw_status not in STATUSES:
                    raise ValueError(f"unknown forward status at item {i}: {raw_status!r}")
                status = STATUSES[raw_status]
                if raw_status == "Hit and Incorrect":
                    result["warnings"].append(f"status_alias:item={i}:Hit and Incorrect")
                # Judge prompts preserve reference order. Never take category labels
                # from the judge when an authoritative benchmark list is available.
                if field(item, "Description", "描述") != field(reference_item, "描述", "Description"):
                    result["warnings"].append(f"description_differs:item={i}:using_reference_order")
                if labels(item) != cats:
                    raise ValueError(f"judge/reference category mismatch at item {i}")
                counts[status] += 1
                for category in cats:
                    categories[category].append(status)
            hallucination = field(judge, "Hallucination", "幻觉")
            summary_h = field(hallucination, "Count", "数量")
            if type(summary_h) is not int or summary_h < 0:
                raise ValueError("hallucination count must be a nonnegative integer")
            extra = field(judge, "Extra_Description_List", "额外描述列表")
            if not isinstance(extra, list):
                raise ValueError("extra description list must be a list")
            list_h = sum(field(x, "Status", "状态") in ("Hallucination", "幻觉") for x in extra)
            result["hallucination_counts"] = dict(summary=summary_h, list=list_h)
            if summary_h != list_h:
                message = f"hallucination_count_conflict:summary={summary_h},list={list_h}"
                if hallu_source == "strict":
                    raise ValueError(message)
                result["warnings"].append(message)
            h = list_h if hallu_source == "list" else summary_h
        result["counts"].update(correct=counts["correct"], incorrect=counts["incorrect"],
                                missed=counts["missed"], hallucinations=h)
        result["metrics"] = metric_values(counts["correct"], counts["incorrect"], n_key, h)
        for category, statuses in categories.items():
            if statuses:
                cc = Counter(statuses)
                result["categories"][category] = dict(
                    n_key=len(statuses), correct=cc["correct"], incorrect=cc["incorrect"],
                    missed=cc["missed"],
                    f1=metric_values(cc["correct"], cc["incorrect"], len(statuses))["f1"],
                )
        result["status"] = "ok"
    except (ValueError, TypeError, SyntaxError, KeyError, AttributeError) as exc:
        result["metrics"] = None
        result["categories"] = {}
        result["errors"].append(str(exc))
    return result


def load_records(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        if path.suffix.lower() == ".csv":
            return list(csv.DictReader(f))
        return [json.loads(line) for line in f if line.strip()]


def index_records(rows):
    indexed = {}
    for row in rows:
        rid = row["relation_id"]
        if not rid or rid in indexed:
            raise ValueError(f"empty or duplicate relation_id: {rid!r}")
        indexed[rid] = row
    return indexed


def aggregate(records, category_average="micro"):
    valid = [r for r in records if r["status"] == "ok"]
    overall = {m: statistics.fmean(r["metrics"][m] for r in valid) if valid else None
               for m in METRICS}
    categories = {}
    if category_average not in ("micro", "macro"):
        raise ValueError("category average must be micro or macro")
    for category in CATEGORIES:
        entries = [r["categories"][category] for r in valid if category in r["categories"]]
        totals = {k: sum(e[k] for e in entries) for k in ("n_key", "correct", "incorrect", "missed")}
        macro = statistics.fmean(e["f1"] for e in entries) if entries else None
        micro = metric_values(totals["correct"], totals["incorrect"], totals["n_key"])["f1"] if entries else None
        categories[category] = dict(
            n_samples=len(entries), n_items=totals["n_key"], counts=totals,
            f1=micro if category_average == "micro" else macro,
            f1_micro=micro, f1_macro=macro,
        )
    return dict(n_expected=len(records), n_scored=len(valid), n_invalid=len(records)-len(valid),
                complete=len(valid)==len(records), overall=overall, categories=categories)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_file(path, dataset, args):
    rows = index_records(load_records(path))
    unexpected = set(rows) - set(dataset)
    if unexpected:
        raise ValueError(f"{path.name}: {len(unexpected)} IDs absent from dataset")
    records = []
    for rid, ref_row in dataset.items():
        if rid not in rows:
            record = dict(status="invalid", errors=["missing_result"], warnings=[],
                          metrics=None, categories={})
        else:
            record = score_sample(rows[rid], ref_row["keypoint_list_with_dimension"],
                                  args.pred_col, args.judge_col, args.hallu_source)
        records.append(dict(relation_id=rid, **record))
    summary = aggregate(records, args.category_average)
    summary.update(model=path.stem, input_file=str(path.resolve()), input_sha256=digest(path),
                   n_input=len(rows))
    return summary, records


def percentage(value):
    return "NA" if value is None else f"{value * 100:.1f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path,
                        help="Saved JSONL/CSV files or a directory containing them")
    parser.add_argument("--dataset", type=Path, default=ROOT / "Data/test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "score_results")
    parser.add_argument("--pred-col", help="Default: prediction_result, then inference_result")
    parser.add_argument("--judge-col", default="judge_result")
    parser.add_argument("--hallu-source", choices=("summary", "list", "strict"), default="summary",
                        help="Default: use Hallucination.Count; report conflicts with the list")
    parser.add_argument("--category-average", choices=("micro", "macro"), default="micro",
                        help="Category F1: pooled item counts (default) or per-pair mean")
    parser.add_argument("--allow-invalid", action="store_true",
                        help="Exit successfully with explicitly incomplete valid-only averages")
    args = parser.parse_args(argv)
    paths = []
    for item in args.input:
        paths.extend(sorted(p for p in item.iterdir() if p.suffix.lower() in (".jsonl", ".csv"))
                     if item.is_dir() else [item])
    if not paths or len({p.stem for p in paths}) != len(paths):
        parser.error("inputs must be nonempty and have unique file stems")
    dataset = index_records(load_records(args.dataset))
    if not dataset:
        parser.error("dataset is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in paths:
        summary, records = run_file(path, dataset, args)
        summaries.append(summary)
        with (args.output_dir / f"{path.stem}.samples.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    report = dict(
        protocol="DiffCap-Bench paper formulas; per-sample macro averages",
        category_protocol=f"{args.category_average} category F1 without extra hallucinations; absent categories excluded",
        invalid_policy="exclude and report; incomplete results are not full-benchmark scores",
        empty_prediction_policy="all metrics zero; included in averages",
        hallucination_source=args.hallu_source,
        dataset_sha256=digest(args.dataset), scorer_sha256=digest(Path(__file__)),
        models=summaries,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    headers = ["Model", "Scored/Expected", "Invalid", *CATEGORIES, *METRICS]
    table = ["All metric values below are percentages. Incomplete runs use valid samples only.", "",
             "|"+"|".join(headers)+"|", "|"+"|".join(["---"]*len(headers))+"|"]
    for s in summaries:
        values = [s["model"], f'{s["n_scored"]}/{s["n_expected"]}', str(s["n_invalid"])]
        values += [percentage(s["categories"][c]["f1"]) for c in CATEGORIES]
        values += [percentage(s["overall"][m]) for m in METRICS]
        table.append("|"+"|".join(values)+"|")
    rendered = "\n".join(table)+"\n"
    (args.output_dir / "summary.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Per-sample diagnostics and summary saved to {args.output_dir}")
    return 2 if any(s["n_invalid"] for s in summaries) and not args.allow_invalid else 0


if __name__ == "__main__":
    sys.exit(main())
