"""Prepare a bucketed mixed SFT JSONL dataset.

This combines clean NuminaMath-CoT reasoning buckets with an oversampled public
MCQ format bucket. Numina rows are selected without duplicates across buckets;
public MCQ rows are intentionally oversampled for boxed-letter discipline.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.supervised_fine_tuning.src.prepare_numina_sft import (
    clean_text,
    looks_bad,
    make_example,
    token_len,
)


SYSTEM_MCQ = (
    "You are an expert mathematician. Pick the best answer letter and put it inside "
    "\\boxed{}, e.g. \\boxed{C}."
)


@dataclass(frozen=True)
class BucketSpec:
    name: str
    n_train: int
    n_val: int
    min_tokens: int
    max_tokens: int
    strategy: str

    @property
    def total(self) -> int:
        return self.n_train + self.n_val


def parse_bucket_spec(raw: str) -> list[BucketSpec]:
    specs: list[BucketSpec] = []
    for part in raw.split(","):
        fields = part.split(":")
        if len(fields) != 6:
            raise ValueError(
                "Each bucket must be name:n_train:n_val:min_tokens:max_tokens:strategy; "
                f"got {part!r}"
            )
        name, n_train, n_val, min_tok, max_tok, strategy = fields
        if strategy not in {"random", "longest"}:
            raise ValueError(f"Unknown bucket strategy {strategy!r} for {name}.")
        spec = BucketSpec(
            name=name,
            n_train=int(n_train),
            n_val=int(n_val),
            min_tokens=int(min_tok),
            max_tokens=int(max_tok),
            strategy=strategy,
        )
        if spec.total <= 0 or spec.min_tokens >= spec.max_tokens:
            raise ValueError(f"Invalid bucket spec: {spec}")
        specs.append(spec)

    sorted_specs = sorted(specs, key=lambda s: (s.min_tokens, s.max_tokens))
    for prev, curr in zip(sorted_specs, sorted_specs[1:]):
        if curr.min_tokens < prev.max_tokens:
            raise ValueError(
                "Bucket token ranges must not overlap so Numina rows cannot duplicate: "
                f"{prev.name} [{prev.min_tokens}, {prev.max_tokens}) and "
                f"{curr.name} [{curr.min_tokens}, {curr.max_tokens})"
            )
    return specs


def count_label(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def format_user_turn(question: str, options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    return f"{question.strip()}\n\nOptions:\n{opts_text}"


def make_public_mcq_example(row: dict[str, Any], tokenizer: Any) -> dict[str, Any] | None:
    options = row.get("options")
    answer = row.get("answer")
    if not isinstance(options, list) or not options:
        return None
    if not isinstance(answer, str) or len(answer.strip()) != 1:
        return None
    answer = answer.strip().upper()
    if not ("A" <= answer <= chr(64 + len(options))):
        return None

    messages = [
        {"role": "system", "content": SYSTEM_MCQ},
        {"role": "user", "content": format_user_turn(str(row.get("question", "")), options)},
        {"role": "assistant", "content": f"Therefore, the answer is \\boxed{{{answer}}}."},
    ]
    return {
        "messages": messages,
        "source": "public_jsonl",
        "kind": "mcq_format",
        "answer": answer,
        "public_id": row.get("id"),
        "n_tokens": token_len(tokenizer, messages),
        "bucket": "public_mcq",
    }


def select_public_mcq(path: Path, tokenizer: Any, n_train: int, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_examples: list[dict[str, Any]] = []
    rejected = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("options"):
                continue
            ex = make_public_mcq_example(row, tokenizer)
            if ex is None:
                rejected += 1
            else:
                base_examples.append(ex)

    if not base_examples and n_train:
        raise RuntimeError(f"No usable public MCQ rows found in {path}.")

    selected: list[dict[str, Any]] = []
    while len(selected) < n_train:
        cycle = list(base_examples)
        rng.shuffle(cycle)
        for ex in cycle:
            selected.append(dict(ex))
            if len(selected) >= n_train:
                break

    return selected, {"usable_base": len(base_examples), "rejected": rejected, "selected_train": len(selected)}


def split_bucket(examples: list[dict[str, Any]], spec: BucketSpec, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(examples) < spec.total:
        raise RuntimeError(
            f"Bucket {spec.name} only has {len(examples)} examples; requested {spec.total}."
        )
    examples = [dict(ex, bucket=spec.name) for ex in examples]
    rng.shuffle(examples)
    return examples[: spec.n_train], examples[spec.n_train : spec.total]


def select_numina_buckets(args: argparse.Namespace, tokenizer: Any, specs: list[BucketSpec], rng: random.Random) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    print(f"Loading {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    print("Columns:", ds.column_names)

    random_seen = {spec.name: 0 for spec in specs if spec.strategy == "random"}
    random_samples: dict[str, list[dict[str, Any]]] = {
        spec.name: [] for spec in specs if spec.strategy == "random"
    }
    longest_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        spec.name: [] for spec in specs if spec.strategy == "longest"
    }
    rejects: dict[str, int] = {
        "looks_bad": 0,
        "make_example": 0,
        "tokenize": 0,
        "no_bucket": 0,
    }

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if args.scan_limit > 0:
        indices = indices[: args.scan_limit]

    by_name = {spec.name: spec for spec in specs}
    heap_counter = 0
    for idx in tqdm(indices, total=len(indices)):
        row = ds[idx]
        problem = clean_text(row.get("problem", ""))
        solution = clean_text(row.get("solution", ""))

        if looks_bad(
            problem,
            solution,
            max_solution_chars=args.max_solution_chars,
            allow_proofs=args.allow_proofs,
        ):
            rejects["looks_bad"] += 1
            continue

        ex = make_example(
            problem,
            solution,
            source=args.dataset_name,
            max_answer_chars=args.max_answer_chars,
            boxed_policy=args.boxed_policy,
        )
        if ex is None:
            rejects["make_example"] += 1
            continue

        try:
            n_tok = token_len(tokenizer, ex["messages"])
        except Exception:
            rejects["tokenize"] += 1
            continue

        ex["n_tokens"] = n_tok
        ex["source_index"] = idx
        matched = False
        for spec in specs:
            if not (spec.min_tokens <= n_tok < spec.max_tokens):
                continue
            matched = True
            if spec.strategy == "random":
                random_seen[spec.name] += 1
                seen = random_seen[spec.name]
                sample = random_samples[spec.name]
                if len(sample) < spec.total:
                    sample.append(ex)
                else:
                    j = rng.randrange(seen)
                    if j < spec.total:
                        sample[j] = ex
            else:
                heap = longest_heaps[spec.name]
                item = (n_tok, heap_counter, ex)
                heap_counter += 1
                if len(heap) < spec.total:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
            break
        if not matched:
            rejects["no_bucket"] += 1

    selected: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        if spec.strategy == "random":
            selected[spec.name] = random_samples[spec.name]
        else:
            selected[spec.name] = [
                item[2] for item in sorted(longest_heaps[spec.name], reverse=True)
            ]

    manifest = {
        "numina_rejects": rejects,
        "random_seen": random_seen,
        "selected_by_bucket": {
            name: {
                "selected": len(rows),
                "requested": by_name[name].total,
                "min_tokens": min((r["n_tokens"] for r in rows), default=None),
                "max_tokens": max((r["n_tokens"] for r in rows), default=None),
            }
            for name, rows in selected.items()
        },
    }
    return selected, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--dataset_name", default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bucket_spec", required=True)
    parser.add_argument("--public_mcq_path", default="data/public.jsonl")
    parser.add_argument("--public_mcq_train", type=int, default=3000)
    parser.add_argument("--max_solution_chars", type=int, default=0)
    parser.add_argument("--max_answer_chars", type=int, default=200)
    parser.add_argument("--boxed_policy", choices=["single", "last"], default="single")
    parser.add_argument("--allow_proofs", action="store_true")
    parser.add_argument("--scan_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    specs = parse_bucket_spec(args.bucket_spec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    selected, manifest = select_numina_buckets(args, tokenizer, specs, rng)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for spec in specs:
        bucket_train, bucket_val = split_bucket(selected[spec.name], spec, rng)
        train_rows.extend(bucket_train)
        val_rows.extend(bucket_val)

    public_rows, public_manifest = select_public_mcq(
        Path(args.public_mcq_path), tokenizer, args.public_mcq_train, rng
    )
    train_rows.extend(public_rows)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_path = out_dir / f"sft_train_{count_label(len(train_rows))}.jsonl"
    val_path = out_dir / f"sft_val_{len(val_rows)}.jsonl"
    manifest_path = out_dir / "manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest.update(
        {
            "args": vars(args),
            "bucket_specs": [asdict(spec) for spec in specs],
            "public_mcq": public_manifest,
            "outputs": {
                "train_path": str(train_path),
                "val_path": str(val_path),
                "manifest_path": str(manifest_path),
            },
            "final_counts": {
                "train": len(train_rows),
                "val": len(val_rows),
                "train_by_bucket": {
                    name: sum(1 for row in train_rows if row.get("bucket") == name)
                    for name in [spec.name for spec in specs] + ["public_mcq"]
                },
                "val_by_bucket": {
                    name: sum(1 for row in val_rows if row.get("bucket") == name)
                    for name in [spec.name for spec in specs]
                },
                "train_token_min": min(row["n_tokens"] for row in train_rows),
                "train_token_max": max(row["n_tokens"] for row in train_rows),
                "val_token_min": min(row["n_tokens"] for row in val_rows),
                "val_token_max": max(row["n_tokens"] for row in val_rows),
            },
        }
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote {len(train_rows)} train examples to {train_path}")
    print(f"Wrote {len(val_rows)} val examples to {val_path}")
    print(f"Wrote manifest to {manifest_path}")
    print("Train by bucket:", manifest["final_counts"]["train_by_bucket"])
    print("Val by bucket:", manifest["final_counts"]["val_by_bucket"])
    print(
        "Train token range:",
        manifest["final_counts"]["train_token_min"],
        manifest["final_counts"]["train_token_max"],
    )


if __name__ == "__main__":
    main()
