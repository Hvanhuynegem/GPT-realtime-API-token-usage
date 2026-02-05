#!/usr/bin/env python3
"""
vqa_soft_accuracy.py

Implements VQA-style strict answer normalization and soft accuracy scoring:

accuracy = min(m/3, 1)
where m = number of human answers (out of 10) that exactly match the model prediction
after normalization.

Inputs:
- prompt (string)                 [not used for scoring, kept for your pipeline]
- baseline_answers (string)       10 human answers, delimited (default: '|')
- received_answer (string)        model predicted answer

Examples:
  python vqa_soft_accuracy.py \
    --prompt "What color is the car?" \
    --baseline "red|red|maroon|red|blue|red|red|red|red|red" \
    --pred "Red."

  python vqa_soft_accuracy.py --interactive
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List, Dict, Tuple


# ----------------------------
# VQA-style normalization
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
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "lets": "let's",
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
    "shed": "she'd",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebody'd",
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

_ARTICLES = {"a", "an", "the"}

# A light number mapping commonly used in VQA eval code:
_NUMBER_MAP = {
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

# Punctuation handling similar to the official VQA evaluator:
# - remove many punctuation marks
# - keep periods in decimals if needed, but generally normalize to whitespace
_PUNCT = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]""")
_MULTI_SPACE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    if s is None:
        return ""

    s = s.strip().lower()

    # Replace some unicode whitespace
    s = s.replace("\u00a0", " ")

    # Remove punctuation by turning into spaces (prevents word-joining)
    s = _PUNCT.sub(" ", s)

    # Collapse whitespace
    s = _MULTI_SPACE.sub(" ", s).strip()

    if not s:
        return ""

    # Token-level processing:
    tokens = s.split()

    # Map written numbers to digits
    tokens = [_NUMBER_MAP.get(t, t) for t in tokens]

    # Remove articles
    tokens = [t for t in tokens if t not in _ARTICLES]

    # Apply contractions normalization (keys are often without apostrophes)
    # We normalize by stripping apostrophes from token to match keys like "dont"
    out_tokens: List[str] = []
    for t in tokens:
        key = t.replace("'", "")
        if key in _CONTRACTIONS:
            out_tokens.append(_CONTRACTIONS[key])
        else:
            out_tokens.append(t)

    s = " ".join(out_tokens)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


# ----------------------------
# Scoring
# ----------------------------

def vqa_soft_accuracy(pred_answer: str, human_answers: List[str]) -> Tuple[float, int, str, List[str]]:
    """
    Returns: (accuracy, m, normalized_pred, normalized_humans)
    """
    norm_pred = _normalize_text(pred_answer)
    norm_humans = [_normalize_text(a) for a in human_answers]

    m = sum(1 for a in norm_humans if a == norm_pred)
    acc = min(m / 3.0, 1.0)
    return acc, m, norm_pred, norm_humans


def _parse_baseline_answers(raw: str, delimiter: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []

    # Allow JSON list input too.
    if raw.startswith("[") and raw.endswith("]"):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except json.JSONDecodeError:
            pass

    # Otherwise, delimiter-separated
    return [part.strip() for part in raw.split(delimiter)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="", help="Question/prompt text (kept for logging).")
    ap.add_argument("--baseline", type=str, default="", help="10 human answers (delimiter-separated) OR JSON list.")
    ap.add_argument("--pred", type=str, default="", help="Model predicted answer.")
    ap.add_argument("--delimiter", type=str, default="|", help="Delimiter for --baseline if not JSON.")
    ap.add_argument("--interactive", action="store_true", help="Read inputs from stdin prompts.")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    args = ap.parse_args()

    if args.interactive:
        prompt = input("Prompt: ").rstrip("\n")
        baseline_raw = input(f"Baseline answers (10, separated by '{args.delimiter}' or JSON list): ").rstrip("\n")
        pred = input("Received answer (prediction): ").rstrip("\n")
    else:
        prompt = args.prompt
        baseline_raw = args.baseline
        pred = args.pred

    human_answers = _parse_baseline_answers(baseline_raw, args.delimiter)
    if len(human_answers) == 0:
        print("Error: baseline answers are empty. Provide 10 human answers.", file=sys.stderr)
        return 2

    acc, m, norm_pred, norm_humans = vqa_soft_accuracy(pred, human_answers)

    result = {
        "prompt": prompt,
        "prediction_raw": pred,
        "prediction_normalized": norm_pred,
        "human_answers_raw": human_answers,
        "human_answers_normalized": norm_humans,
        "matches_m": m,
        "accuracy": acc,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Prompt: {prompt}")
        print(f"Prediction (raw): {pred}")
        print(f"Prediction (normalized): {norm_pred}")
        print(f"Human answers (raw): {human_answers}")
        print(f"Human answers (normalized): {norm_humans}")
        print(f"Matches m: {m} / {len(human_answers)}")
        print(f"Soft accuracy = min(m/3, 1) = {acc:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
