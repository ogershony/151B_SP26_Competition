"""Generate private-set submission JSONL.

This intentionally runs as a CLI so notebooks can launch it in a subprocess.
vLLM is more reliable from a normal Python process than from an ipykernel cell,
where stdout/CUDA multiprocessing can break engine startup.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("REPO_ROOT", str(_REPO_ROOT))

from experiments.prompt_engineering.src.eval import _load_named_regime  # noqa: E402
from experiments.prompt_engineering.src.prompts import PROMPTS  # noqa: E402
from shared.io import load_jsonl, save_jsonl  # noqa: E402
from shared.prompt_format import build_chat_messages  # noqa: E402
from shared.runner import load_model  # noqa: E402
from shared.schemas import RunnerCfg  # noqa: E402
from shared.scoring import _extract_letter, _last_boxed_content  # noqa: E402
from shared.voting import vote_index  # noqa: E402


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def _extract_private_answer(item: dict, response: str) -> str:
    if item.get("options"):
        return _extract_letter(response)
    return _last_boxed_content(response).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt_id", default="cot_structured_sc")
    parser.add_argument("--regime", default="high_div_8")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--slice_indices", default=None)
    parser.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--quant", default="bf16", choices=["bf16", "bnb"])
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    private_path = Path(args.private_path)
    output_path = Path(args.output_path)
    if not private_path.exists():
        raise FileNotFoundError(f"Private data not found: {private_path}")
    if private_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite the private input file.")

    items = load_jsonl(private_path)
    slice_indices = _parse_indices(args.slice_indices)
    if slice_indices is not None:
        items = [items[i] for i in slice_indices if 0 <= i < len(items)]
    if not items:
        raise ValueError(f"No private items loaded from {private_path}.")

    completed: dict[int, dict] = {}
    if not args.no_resume and output_path.exists():
        for row in load_jsonl(output_path):
            if "id" in row and "response" in row:
                completed[int(row["id"])] = row
        print(f"Resuming from {output_path}: {len(completed)} completed rows found.")

    if args.prompt_id not in PROMPTS:
        raise KeyError(f"Unknown prompt_id {args.prompt_id!r}. Known: {sorted(PROMPTS)}")
    prompt = PROMPTS[args.prompt_id]
    sampling = _load_named_regime(args.regime)
    if args.n_samples is not None:
        sampling = replace(sampling, name=f"{sampling.name}_n{args.n_samples}", n_samples=args.n_samples)

    adapter_path = None if args.adapter_path in {None, "", "null", "None"} else args.adapter_path
    runner_cfg = RunnerCfg(engine=args.engine, quant=args.quant, adapter_path=adapter_path)

    print(f"Loaded {len(items)} private items from {private_path}")
    print(
        f"Prompt={prompt.id}, regime={sampling.name}, n_samples={sampling.n_samples}, "
        f"max_tokens={args.max_tokens}, batch_size={args.batch_size}"
    )
    print(
        "Runner="
        f"engine={runner_cfg.engine}, quant={runner_cfg.quant}, adapter_path={runner_cfg.adapter_path}"
    )

    print("Loading model...")
    handle = load_model(runner_cfg)
    print("Model loaded.")

    pending = [item for item in items if int(item["id"]) not in completed]
    batch_size = max(1, int(args.batch_size))
    total_chunks = (len(pending) + batch_size - 1) // batch_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating private responses: {len(pending)} pending / {len(items)} total")

    start_time = time.time()
    for chunk_idx, start in enumerate(range(0, len(pending), batch_size), start=1):
        chunk = pending[start : start + batch_size]
        print(f"Chunk {chunk_idx}/{total_chunks}: generating {len(chunk)} items...")
        messages = [build_chat_messages(item, prompt) for item in chunk]
        outputs = handle.generate_batch(messages, sampling, args.max_tokens)
        for item, out in zip(chunk, outputs):
            extracts = [_extract_private_answer(item, response) for response in out.responses]
            idx = vote_index(extracts)
            completed[int(item["id"])] = {
                "id": int(item["id"]),
                "is_mcq": bool(item.get("options")),
                "response": out.responses[idx],
            }
        rows = [completed[int(item["id"])] for item in items if int(item["id"]) in completed]
        save_jsonl(rows, output_path)
        elapsed_min = (time.time() - start_time) / 60.0
        print(f"  saved {len(rows)}/{len(items)} rows to {output_path} (elapsed={elapsed_min:.1f} min)")

    rows = [completed[int(item["id"])] for item in items if int(item["id"]) in completed]
    save_jsonl(rows, output_path)
    print(f"Saved {len(rows)} private submission rows to {output_path}")
    print(f"Elapsed: {(time.time() - start_time) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
