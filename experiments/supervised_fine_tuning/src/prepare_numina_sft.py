"""Prepare NuminaMath-CoT chat JSONL for QLoRA SFT.

The output format matches ``train_qlora_sft.py``: each row has a ``messages``
list ending in an assistant completion. Token filtering uses the same Qwen chat
template as training, so ``--max_tokens`` should usually match
``train_qlora_sft.py --max_seq_len``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


SYSTEM_FREE = (
    "You are an expert mathematician. Solve the problem clearly and efficiently. "
    "Put the final answer inside \\boxed{}."
)


def clean_text(text: Any) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_boxed_answers(text: str) -> list[str]:
    """Find balanced-brace ``\\boxed{...}`` answers."""
    out: list[str] = []
    i = 0
    marker = r"\boxed{"

    while True:
        start = text.find(marker, i)
        if start == -1:
            break

        j = start + len(marker)
        depth = 1
        ans_chars: list[str] = []

        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
                ans_chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth > 0:
                    ans_chars.append(ch)
            else:
                ans_chars.append(ch)
            j += 1

        if depth == 0:
            ans = "".join(ans_chars).strip()
            if ans:
                out.append(ans)

        i = j

    return out


def remove_last_boxed_region(text: str) -> str:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return text

    j = start + len(marker)
    depth = 1
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1

    if depth == 0:
        return clean_text(text[:start] + text[j:])
    return text


def looks_bad(
    problem: str,
    solution: str,
    *,
    max_solution_chars: int,
    allow_proofs: bool,
) -> bool:
    problem_l = problem.lower()
    solution_l = solution.lower()
    joined = problem_l + "\n" + solution_l

    bad_markers = [
        "shown in the figure",
        "shown below",
        "in the diagram",
        "from the diagram",
        "see figure",
        "use python",
        "using python",
        "write a program",
        "code",
        "calculator",
    ]
    if any(x in joined for x in bad_markers):
        return True

    if len(problem) < 20 or len(solution) < 50:
        return True

    if max_solution_chars > 0 and len(solution) > max_solution_chars:
        return True

    if not allow_proofs:
        proof_markers = ["prove that", "show that", "establish that"]
        if any(x in problem_l for x in proof_markers):
            return True

    return False


def build_assistant_trace(solution: str, answer: str) -> str:
    solution = clean_text(solution)
    reasoning = remove_last_boxed_region(solution)
    reasoning = re.sub(
        r"(therefore|thus|hence),?\s+the\s+answer\s+is\s*\.?$",
        "",
        reasoning,
        flags=re.IGNORECASE,
    ).strip()

    return (
        "<think>\n"
        f"{reasoning}\n"
        "</think>\n\n"
        f"Therefore, the answer is \\boxed{{{answer}}}."
    )


def is_multipart_problem(problem: str) -> bool:
    p = problem.lower()
    patterns = [
        r"\(\s*1\s*\)",
        r"\(\s*2\s*\)",
        r"\(\s*i\s*\)",
        r"\(\s*ii\s*\)",
        r"part\s+a",
        r"part\s+b",
        r"\bfind:\s*",
        r"find the following",
    ]
    hits = sum(1 for pat in patterns if re.search(pat, p))
    return hits >= 2


def has_bad_blank_artifacts(text: str) -> bool:
    bad_patterns = [
        r"\$\s*\$",
        r"\$\$\s*\$\$",
        r"=\s*\$\$\s*[\.\n]",
        r"answer is\s*\$\$",
        r"is\s*\$\$\s+",
        r"therefore.*\$\$",
    ]
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in bad_patterns)


def answer_is_suspicious(answer: str, *, max_answer_chars: int) -> bool:
    a = answer.strip()
    if not a:
        return True
    if len(a) > max_answer_chars:
        return True
    if re.search(r"\d+\.\d{8,}", a):
        return True
    return False


def make_example(
    problem: str,
    solution: str,
    *,
    source: str,
    max_answer_chars: int,
    boxed_policy: str,
) -> dict[str, Any] | None:
    problem = clean_text(problem)
    solution = clean_text(solution)
    boxed = find_boxed_answers(solution)

    if boxed_policy == "single" and len(boxed) != 1:
        return None
    if boxed_policy == "last" and not boxed:
        return None

    answer = boxed[-1].strip()
    if answer_is_suspicious(answer, max_answer_chars=max_answer_chars):
        return None
    if is_multipart_problem(problem):
        return None
    if has_bad_blank_artifacts(solution):
        return None

    assistant = build_assistant_trace(solution, answer)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_FREE},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "kind": "free_response",
        "answer": answer,
    }


def token_len(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--dataset_name", default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--out_dir", default="data_numina_32k")
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--min_tokens", type=int, default=128)
    parser.add_argument(
        "--max_solution_chars",
        type=int,
        default=0,
        help="Reject raw solutions longer than this many chars. 0 disables this gate.",
    )
    parser.add_argument("--max_answer_chars", type=int, default=200)
    parser.add_argument(
        "--boxed_policy",
        choices=["single", "last"],
        default="single",
        help="single requires exactly one boxed answer; last uses the final boxed answer.",
    )
    parser.add_argument("--allow_proofs", action="store_true")
    parser.add_argument(
        "--selection_strategy",
        choices=["random", "longest"],
        default="random",
        help=(
            "random reproduces the original shuffled-first-N behavior; longest "
            "scans candidates and keeps the longest rows under --max_tokens."
        ),
    )
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=0,
        help="Maximum shuffled source rows to scan before selecting. 0 scans all rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading {args.dataset_name} ({args.dataset_split})...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    print("Columns:", ds.column_names)

    indices = list(range(len(ds)))
    random.shuffle(indices)

    kept: list[dict[str, Any]] = []
    target_total = args.n_train + args.n_val
    rejects: dict[str, int] = {
        "looks_bad": 0,
        "make_example": 0,
        "tokenize": 0,
        "too_short": 0,
        "too_long": 0,
    }

    if args.scan_limit > 0:
        indices = indices[: args.scan_limit]

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

        if n_tok < args.min_tokens:
            rejects["too_short"] += 1
            continue
        if n_tok > args.max_tokens:
            rejects["too_long"] += 1
            continue

        ex["n_tokens"] = n_tok
        kept.append(ex)
        if args.selection_strategy == "random" and len(kept) >= target_total:
            break

    if len(kept) < target_total:
        raise RuntimeError(
            f"Only found {len(kept)} usable examples; requested {target_total}. "
            f"Reject counts: {rejects}"
        )

    if args.selection_strategy == "longest":
        kept.sort(key=lambda x: x["n_tokens"], reverse=True)
        kept = kept[:target_total]

    random.shuffle(kept)
    train = kept[: args.n_train]
    val = kept[args.n_train : args.n_train + args.n_val]

    train_path = out_dir / f"sft_train_{args.n_train // 1000}k.jsonl"
    val_path = out_dir / f"sft_val_{args.n_val}.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(train)} train examples to {train_path}")
    print(f"Wrote {len(val)} val examples to {val_path}")
    print(f"Reject counts: {rejects}")

    if train:
        lengths = [x["n_tokens"] for x in train]
        print(f"Average train tokens: {sum(lengths) / len(lengths):.1f}")
        print(f"Min train tokens: {min(lengths)}")
        print(f"Max train tokens: {max(lengths)}")

    preview_path = out_dir / "preview_first_example.json"
    if train:
        with preview_path.open("w", encoding="utf-8") as f:
            json.dump(train[0], f, indent=2, ensure_ascii=False)
        print(f"Preview example saved to {preview_path}")


if __name__ == "__main__":
    main()
