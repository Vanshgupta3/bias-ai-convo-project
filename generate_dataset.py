"""
Generates labeled cognitive-bias training data via the Groq API (free tier)
and writes it to a CSV matching your existing dataset format (text, label).

Get a free API key at: https://console.groq.com/keys

Usage:
    export GROQ_API_KEY=your_key_here
    python generate_dataset.py --per_class 80 --out dataset.csv
"""

import os
import csv
import json
import time
import argparse
from groq import Groq

LABELS = [
    "sunk_cost",
    "overgeneralization",
    "bandwagon",
    "confirmation_bias",
    "fundamental_attribution",
    "overconfidence",
    "hindsight",
    "availability",
    "no_bias",
]

LABEL_DESCRIPTIONS = {
    "sunk_cost": "continuing something because of past investment (time/money/effort) rather than future value",
    "overgeneralization": "drawing a broad, sweeping conclusion from one or few instances",
    "bandwagon": "believing/doing something because many other people do",
    "confirmation_bias": "only seeking or trusting info that confirms existing beliefs, ignoring contrary evidence",
    "fundamental_attribution": "attributing someone's behavior to their character/nature rather than their situation",
    "overconfidence": "overestimating the correctness or certainty of one's own judgment",
    "hindsight": "believing an outcome was predictable after it already happened ('I knew it all along')",
    "availability": "judging likelihood/importance of something based on how easily examples come to mind (e.g. recent news, vivid memories)",
    "no_bias": "a normal, reasonable statement with no cognitive bias present",
}

PROMPT_TEMPLATE = """Generate {n} diverse example sentences of a person expressing the cognitive bias: "{label}".

Definition: {desc}

Requirements:
- Vary sentence length (short and long), tone (casual, formal, frustrated, calm), and phrasing.
- Include some subtle/borderline examples, not just obvious ones.
- Do NOT use the same sentence template repeatedly.
- For "no_bias", write reasonable, level-headed statements that might superficially resemble one of the other categories but aren't biased.
- Return ONLY a JSON array of strings, nothing else. No markdown, no preamble.

Example output format:
["sentence one", "sentence two", ...]
"""


def generate_for_label(client, model_name, label, n, batch_size=20):
    examples = []
    fails = 0
    while len(examples) < n:
        batch_n = min(batch_size, n - len(examples))
        prompt = PROMPT_TEMPLATE.format(n=batch_n, label=label, desc=LABEL_DESCRIPTIONS[label])
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [warn] request failed for {label}: {e}")
            time.sleep(3)
            fails += 1
            if fails > 5:
                print(f"  [error] too many failures for {label}, moving on")
                break
            continue
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            batch = json.loads(text)
        except json.JSONDecodeError:
            print(f"  [warn] failed to parse batch for {label}, skipping batch")
            continue
        examples.extend(batch)
        print(f"  {label}: {len(examples)}/{n}")
        # keep requests spaced out to be gentle on free tier limits
        time.sleep(2)
    return examples[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=80)
    parser.add_argument("--out", type=str, default="dataset.csv")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable first.")

    client = Groq(api_key=api_key)

    rows = []
    for label in LABELS:
        print(f"Generating for: {label}")
        examples = generate_for_label(client, args.model, label, args.per_class)
        for ex in examples:
            rows.append({"text": ex, "label": label})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()