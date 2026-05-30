"""GRPO RL training for Qwen3-4B-Thinking, using the answer-checker as the reward.

The whole loop is intentionally small and readable:

  1. Build a prompt dataset from `data/public.jsonl[start:]` (disjoint from the
     first-100 eval slice — no leakage). Each row's `prompt` is the SAME chat-message
     format the eval and prompt-engineering tracks use (`build_chat_messages` +
     `BASELINE_STARTER`), so RL trains on the exact distribution we evaluate on.
  2. For each prompt, TRL's `GRPOTrainer` samples a *group* of `num_generations`
     completions, scores each with the reward, and nudges the policy toward the
     above-average ones (GRPO = group-relative advantages, no learned critic).
  3. The reward (see `reward.py`) is correctness + a small shaping term. Correctness
     IS the eval scorer (`shared.scoring.score_one(...).correct` → 1.0/0.0), so training
     and eval never disagree on "correct"; the shaping adds a bounded brevity/format/
     truncation signal that keeps all-correct groups from collapsing to zero advantage.

bf16 weights + a fresh LoRA adapter (or warm-started from an SFT adapter). Rollouts
use vLLM by default and fall back to HF generation if vLLM is unavailable.

Run:
  REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.train_grpo grpo=smoke run_name=grpo_smoke
  REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.train_grpo grpo=demo  run_name=grpo_demo_v1
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Make the repo root importable so `shared.*` resolves, and export REPO_ROOT for
# the Hydra `searchpath` interpolation in config.yaml — derived from the file
# location so it works regardless of cwd (e.g. Colab running from /content).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("REPO_ROOT", str(_REPO_ROOT))

# vLLM colocate + the GRPO logits chain share one GPU; the large [B, L, vocab]
# allocations fragment the caching allocator. Expandable segments lets freed blocks
# be reused across differently-sized allocs, which cuts OOMs on the 80GB A100. Must
# be set before torch initialises CUDA, hence module-load. setdefault → overridable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from shared.io import load_jsonl  # noqa: E402
from shared.prompt_format import build_chat_messages  # noqa: E402
from shared.prompts import BASELINE_STARTER  # noqa: E402

from experiments.reinforcement_learning.src.reward import build_reward_funcs  # noqa: E402

logger = logging.getLogger(__name__)


# ─── Dataset ─────────────────────────────────────────────────────────────────


def build_dataset(data_path: str, start_index: int, max_train_samples: int | None):
    """Turn `public.jsonl[start_index:]` into a conversational GRPO dataset.

    Columns:
      - `prompt`        chat messages (handles MCQ vs free-form system prompt + options)
      - `answer_json`   gold answer, JSON-encoded
      - `options_json`  options list (or null), JSON-encoded
      - `id`            question id (for logging)

    `answer` is sometimes a `str` (MCQ) and sometimes a `list` (free-form), and
    `options` is present only for MCQ. JSON-encoding both sidesteps Arrow's
    nested-type unification, which would otherwise silently corrupt or reject the
    mixed-type columns. The reward decodes them back before scoring.
    """
    from datasets import Dataset

    path = Path(data_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    rows = load_jsonl(path)[start_index:]
    if max_train_samples:
        rows = rows[:max_train_samples]

    records = [
        {
            "prompt": build_chat_messages(item, BASELINE_STARTER),
            "answer_json": json.dumps(item["answer"]),
            "options_json": json.dumps(item.get("options")),
            "id": int(item["id"]),
        }
        for item in rows
    ]
    logger.info("Built GRPO dataset: %d prompts (from index %d).", len(records), start_index)
    return Dataset.from_list(records)


# ─── Reward ──────────────────────────────────────────────────────────────────
# The reward funcs live in `reward.py` (so they're unit-testable without TRL/GPU
# — run `python -m experiments.reinforcement_learning.src.reward` for the
# evaluation harness). `build_reward_funcs` returns a `[correctness, shaping]`
# pair: pure 0/1 correctness plus a small brevity/format/truncation shaping term.
# The shaping is what keeps all-correct groups from collapsing to zero advantage.


# ─── Model + LoRA ──────────────────────────────────────────────────────────


def load_model_and_lora(cfg: DictConfig):
    """bf16 base model (no quantization) + a fresh or warm-started LoRA adapter.

    Returns `(model, tokenizer, peft_config)`. When warm-starting from an existing
    adapter we attach it here and return `peft_config=None`; otherwise we return a
    `LoraConfig` for `GRPOTrainer` to apply itself (do NOT also `get_peft_model`).
    """
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg.model.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    peft_config = None
    init_adapter = cfg.model.init_adapter_path
    if init_adapter:
        logger.info("Warm-starting from adapter %s", init_adapter)
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
            target_modules=list(cfg.lora.target_modules),
        )
    return model, tokenizer, peft_config


def build_grpo_config(cfg: DictConfig, output_dir: Path, use_vllm: bool):
    from trl import GRPOConfig

    g = cfg.grpo
    return GRPOConfig(
        output_dir=str(output_dir),
        seed=cfg.seed,
        # generation / group
        num_generations=g.num_generations,
        max_prompt_length=g.max_prompt_length,
        max_completion_length=g.max_completion_length,
        temperature=g.temperature,
        top_p=g.top_p,
        # RL / optimization
        beta=g.beta,
        learning_rate=g.learning_rate,
        per_device_train_batch_size=g.per_device_train_batch_size,
        gradient_accumulation_steps=g.gradient_accumulation_steps,
        max_steps=g.max_steps,
        num_train_epochs=g.num_train_epochs,
        warmup_ratio=g.warmup_ratio,
        lr_scheduler_type=g.lr_scheduler_type,
        bf16=g.bf16,
        gradient_checkpointing=g.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # rollout backend
        use_vllm=use_vllm,
        vllm_mode=g.vllm_mode,
        vllm_gpu_memory_utilization=g.vllm_gpu_memory_utilization,
        # logging
        log_completions=g.log_completions,
        logging_steps=g.logging_steps,
        save_steps=g.save_steps,
        report_to=g.report_to,
    )


def _ensure_distributed_env() -> None:
    """Make vLLM colocate rollouts work under a plain `python -m` launch.

    TRL's colocate path builds the vLLM engine with
    `distributed_executor_backend="external_launcher"`, and that backend reads
    `RANK` (+ `WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`/`LOCAL_RANK`) from the env and
    needs a `torch.distributed` group. Under `python -m` (as the Colab notebook runs)
    none are set, so vLLM raises `KeyError: 'RANK'` and we fall back to slow HF
    generation. Populate the single-process values — byte-for-byte what
    `torchrun --nproc_per_node=1` would set — so accelerate initialises a 1-rank group
    and colocate works without a launcher. `setdefault` makes this a no-op under a real
    torchrun/accelerate launch, which set these first.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def _train_once(cfg: DictConfig, dataset, output_dir: Path, use_vllm: bool):
    """Build the model + trainer fresh and train. Fresh build per attempt keeps the
    vLLM→HF fallback clean (no double-wrapped LoRA from a half-built trainer)."""
    from trl import GRPOTrainer

    model, tokenizer, peft_config = load_model_and_lora(cfg)
    args = build_grpo_config(cfg, output_dir, use_vllm=use_vllm)
    g = cfg.grpo
    reward_funcs = build_reward_funcs(
        max_completion_length=g.max_completion_length,
        brevity_weight=g.reward_brevity_weight,
        brevity_soft_len=g.reward_brevity_soft_len,  # null → defaults to max_completion_length
        format_weight=g.reward_format_weight,
        trunc_weight=g.reward_trunc_weight,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    return trainer, tokenizer


# ─── Main ────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_name = cfg.run_name or f"grpo_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"
    output_dir = (_REPO_ROOT / cfg.results_dir / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "train_config.yaml")
    logger.info("GRPO run '%s' → %s", run_name, output_dir)

    dataset = build_dataset(
        cfg.grpo.train_data_path,
        cfg.grpo.train_start_index,
        cfg.grpo.max_train_samples,
    )

    if cfg.grpo.use_vllm:
        _ensure_distributed_env()

    try:
        trainer, tokenizer = _train_once(cfg, dataset, output_dir, use_vllm=cfg.grpo.use_vllm)
    except Exception as e:  # noqa: BLE001
        if not cfg.grpo.use_vllm:
            raise
        logger.warning(
            "vLLM rollout path failed (%s: %s). Falling back to HF generation and retrying.",
            type(e).__name__, e,
        )
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        trainer, tokenizer = _train_once(cfg, dataset, output_dir, use_vllm=False)

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    logger.info("Saved final adapter to %s", final_adapter_dir)

    print(f"\nGRPO training done. Adapter: {final_adapter_dir}")
    print(
        "Evaluate on the first-100 slice:\n"
        f"  REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.eval "
        f"adapter_path={final_adapter_dir} run_name={run_name}_eval"
    )


if __name__ == "__main__":
    main()
