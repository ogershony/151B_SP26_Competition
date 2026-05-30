"""Standalone eval for a GRPO-trained adapter on the standard first-100 slice.

Deliberately small and self-contained: it reuses the shared eval primitives
(`load_eval_slice`, `build_chat_messages`, `load_model`, `score_one`, `ResultRow`)
but does NOT route through the prompt-engineering sweep harness. Greedy decoding
keeps the numbers deterministic and comparable across tracks.

  # baseline (base model, no adapter)
  REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.eval run_name=grpo_base_eval
  # a trained adapter
  REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.eval \
    adapter_path=experiments/reinforcement_learning/results/grpo_demo_v1/final_adapter \
    run_name=grpo_demo_v1_eval
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("REPO_ROOT", str(_REPO_ROOT))

from shared.data import load_eval_slice  # noqa: E402
from shared.io import ResultRow, save_jsonl  # noqa: E402
from shared.prompt_format import build_chat_messages  # noqa: E402
from shared.prompts import BASELINE_STARTER  # noqa: E402
from shared.runner import load_model  # noqa: E402
from shared.schemas import EvalSliceCfg, RunnerCfg, SamplingCfg  # noqa: E402
from shared.scoring import score_one  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    eval_cfg = EvalSliceCfg(slice=cfg.eval.slice, max_tokens=cfg.eval.max_tokens, data_path=cfg.eval.data_path)
    items = load_eval_slice(eval_cfg)

    # `adapter_path` (top-level) overrides runner.adapter_path; load_model also honors $ADAPTER_PATH.
    adapter_path = cfg.adapter_path or cfg.runner.adapter_path
    runner = RunnerCfg(engine=cfg.runner.engine, quant=cfg.runner.quant, adapter_path=adapter_path)
    handle = load_model(runner)

    sampling = SamplingCfg(name="greedy", temperature=0.0)
    messages = [build_chat_messages(item, BASELINE_STARTER) for item in items]
    outputs = handle.generate_batch(messages, sampling, eval_cfg.max_tokens)

    rows: list[ResultRow] = []
    n_correct = mcq_total = mcq_correct = free_total = free_correct = 0
    for item, out in zip(items, outputs):
        response = out.responses[0]
        scored = score_one(item, response)
        is_mcq = bool(item.get("options"))
        n_correct += scored.correct
        if is_mcq:
            mcq_total += 1
            mcq_correct += scored.correct
        else:
            free_total += 1
            free_correct += scored.correct
        rows.append(ResultRow(
            id=int(item["id"]), run_id=cfg.run_name, prompt_id="baseline_starter",
            is_mcq=is_mcq, gold=item["answer"], response=response,
            extracted=scored.extracted, correct=scored.correct,
            finished_with_box=scored.finished_with_box,
            n_response_tokens=out.n_response_tokens[0], n_iters=1,
            sampling={"temperature": 0.0}, max_tokens=eval_cfg.max_tokens,
        ))

    out_path = (_REPO_ROOT / cfg.results_dir / f"{cfg.run_name}.jsonl").resolve()
    save_jsonl(rows, out_path)

    n = len(items)
    pct = lambda c, t: f"{100 * c / t:.1f}%" if t else "n/a"  # noqa: E731
    print(f"\nadapter   : {adapter_path or '(base model)'}")
    print(f"overall   : {pct(n_correct, n)}  ({n_correct}/{n})")
    print(f"MCQ       : {pct(mcq_correct, mcq_total)}  ({mcq_correct}/{mcq_total})")
    print(f"free-form : {pct(free_correct, free_total)}  ({free_correct}/{free_total})")
    print(f"results   : {out_path}")


if __name__ == "__main__":
    main()
