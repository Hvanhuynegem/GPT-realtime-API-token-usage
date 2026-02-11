#!/usr/bin/env python3
"""
vqa_soft_accuracy.py

Batch-evaluates your responses.jsonl against prepared.json using the official VQA eval logic
mirroring the VQAEval code you provided.

Inputs (defaults set to your paths):
- dataset_json:   data/VQA-HMUG-data/prepared.json
- responses_jsonl: logs/<run_folder>/responses.jsonl

Outputs:
- metrics printed to stdout
- optional JSON report written to disk

Example:
  python vqa_soft_accuracy.py
  python vqa_soft_accuracy.py --responses logs/2026-02-04_14-01-39_GPT_Realtime_Mini/responses.jsonl
  python vqa_soft_accuracy.py --out logs/2026-02-04_14-01-39_GPT_Realtime_Mini/vqa_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable


# ----------------------------
# Official VQA normalization constants (mirrors VQAEval)
# ----------------------------

_CONTRACTIONS: Dict[str, str] = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id've": "i'd've",
    "i'dve": "i'd've",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

_MANUAL_MAP: Dict[str, str] = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_ARTICLES = {"a", "an", "the"}

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT_LIST = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def _preclean(s: str) -> str:
    if s is None:
        return ""
    return s.replace("\n", " ").replace("\t", " ").strip()


def _process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in _PUNCT_LIST:
        if (p + " " in in_text or " " + p in in_text) or (re.search(_COMMA_STRIP, in_text) is not None):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    out_text = _PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def _process_digit_article(in_text: str) -> str:
    out_words: List[str] = []
    temp_text = in_text.lower().split()
    for w in temp_text:
        w = _MANUAL_MAP.setdefault(w, w)
        if w not in _ARTICLES:
            out_words.append(w)

    for i, w in enumerate(out_words):
        if w in _CONTRACTIONS:
            out_words[i] = _CONTRACTIONS[w]

    return " ".join(out_words)


def _maybe_normalize_for_vqa(gt_answers: List[str], pred_answer: str) -> Tuple[List[str], str, bool]:
    gt_clean = [_preclean(a) for a in gt_answers]
    pred_clean = _preclean(pred_answer)

    do_norm = len(set(gt_clean)) > 1
    if not do_norm:
        return gt_clean, pred_clean, False

    gt_norm = [_process_digit_article(_process_punctuation(a)) for a in gt_clean]
    pred_norm = _process_digit_article(_process_punctuation(pred_clean))
    return gt_norm, pred_norm, True


def vqa_official_accuracy(pred_answer: str, human_answers: List[str]) -> Tuple[float, int, str, List[str], bool]:
    humans_used, pred_used, used_norm = _maybe_normalize_for_vqa(human_answers, pred_answer)

    m_total = sum(1 for a in humans_used if a == pred_used)

    gt_acc: List[float] = []
    for i in range(len(humans_used)):
        other = [humans_used[j] for j in range(len(humans_used)) if j != i]
        matching = sum(1 for a in other if a == pred_used)
        gt_acc.append(min(1.0, float(matching) / 3.0))

    acc = float(sum(gt_acc)) / float(len(gt_acc)) if gt_acc else 0.0
    return acc, m_total, pred_used, humans_used, used_norm


# ----------------------------
# I/O helpers for your files
# ----------------------------

def _load_prepared_dataset(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in data:
        sid = str(item.get("id"))
        by_id[sid] = item
    return by_id


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=r"data\VQA-HMUG-data\prepared.json",
        help="Path to prepared.json",
    )
    ap.add_argument(
        "--responses",
        type=str,
        default=r"logs\2026-02-10_16-20-05_GPT_Realtime\responses.jsonl",
        help="Path to responses.jsonl",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=r"logs\2026-02-10_16-20-05_GPT_Realtime\vqa_eval.json",
        help="Optional path to write a JSON report (e.g., logs/.../vqa_eval.json).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary to stdout.",
    )
    ap.add_argument(
        "--per-preprocessor",
        action="store_true",
        help="Also compute and print per-preprocessor summary metrics.",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    responses_path = Path(args.responses)

    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    if not responses_path.exists():
        print(f"Error: responses not found: {responses_path}", file=sys.stderr)
        return 2

    dataset_by_id = _load_prepared_dataset(dataset_path)

    # Per-preprocessor accumulators
    acc_sum: Dict[str, float] = defaultdict(float)
    count: Dict[str, int] = defaultdict(int)

    # Detailed per-line results (optional for output file)
    detailed: List[Dict[str, Any]] = []

    missing_in_dataset = 0
    total_lines = 0

    for row in _iter_jsonl(responses_path):
        total_lines += 1

        preproc = str(row.get("preprocessor", "UNKNOWN"))
        sample_id = str(row.get("sample_id", ""))
        pred = str(row.get("assistant_text", ""))

        item = dataset_by_id.get(sample_id)
        if item is None:
            missing_in_dataset += 1
            continue

        prompt = str(item.get("question", ""))
        human_answers = item.get("answers", [])
        if not isinstance(human_answers, list) or len(human_answers) == 0:
            continue

        acc, m_total, pred_used, humans_used, used_norm = vqa_official_accuracy(pred, [str(a) for a in human_answers])

        acc_sum[preproc] += acc
        count[preproc] += 1

        if args.out:
            detailed.append(
                {
                    "preprocessor": preproc,
                    "sample_id": sample_id,
                    "question": prompt,
                    "prediction_raw": pred,
                    "prediction_used_for_eval": pred_used,
                    "human_answers_used_for_eval": humans_used,
                    "used_normalization": used_norm,
                    "matches_m_total": m_total,
                    "accuracy_official_vqa": acc,
                }
            )

    # Summary
    per_preproc = {}
    overall_acc_sum = 0.0
    overall_n = 0

    for p in sorted(count.keys()):
        n = count[p]
        a = acc_sum[p] / n if n else 0.0
        per_preproc[p] = {"n": n, "accuracy": a}
        overall_acc_sum += acc_sum[p]
        overall_n += n

    overall_acc = overall_acc_sum / overall_n if overall_n else 0.0

    summary = {
        "dataset": str(dataset_path),
        "responses": str(responses_path),
        "total_response_lines": total_lines,
        "matched_predictions": overall_n,
        "missing_in_dataset": missing_in_dataset,
        "overall_accuracy_official_vqa": overall_acc,
        "per_preprocessor": per_preproc,
    }

    # Write report if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_obj = {"summary": summary, "details": detailed}
        out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Dataset:   {dataset_path}")
        print(f"Responses: {responses_path}")
        print(f"Matched predictions: {overall_n} (missing_in_dataset={missing_in_dataset}, total_lines={total_lines})")
        print(f"Overall accuracy (official VQA): {overall_acc:.4f}")

        if args.per_preprocessor:
            print("\nPer-preprocessor:")
            for p in sorted(per_preproc.keys()):
                print(f"  {p}: n={per_preproc[p]['n']}, acc={per_preproc[p]['accuracy']:.4f}")

        if args.out:
            print(f"\nWrote report: {Path(args.out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
