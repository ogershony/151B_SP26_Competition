"""Hydra-driven eval CLI for the prompt-engineering experiment.

Loads the model once and dispatches each entry in `cfg.run.runs` by prompt
`kind` (single / self_consistency / php / self_refine). Per-prompt JSONL lands
under `experiments/prompt_engineering/results/`; a leaderboard prints at the end.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Make the repo root importable so `shared.*` resolves whether we're run as a
# module or as a script. Also export REPO_ROOT for the Hydra `searchpath`
# interpolation in config.yaml — derived from the file location so it works
# regardless of the user's cwd (e.g. Colab running from /content).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("REPO_ROOT", str(_REPO_ROOT))

from shared.data import load_eval_slice  # noqa: E402
from shared.io import ResultRow, save_jsonl  # noqa: E402
from shared.multi_turn import run_php, run_self_refine  # noqa: E402
from shared.prompt_format import build_chat_messages  # noqa: E402
from shared.prompts import Prompt  # noqa: E402
from shared.runner import MODEL_ID, ModelHandle, load_model, resolve_adapter_path  # noqa: E402
from shared.schemas import (  # noqa: E402
    EvalSliceCfg,
    PromptRunEntryCfg,
    RunnerCfg,
    SamplingCfg,
    register_configs,
)
from shared.scoring import score_one  # noqa: E402
from shared.telemetry import Timer, TimingsRegistry, use_registry  # noqa: E402
from shared.voting import vote_index  # noqa: E402

from experiments.prompt_engineering.src.prompts import PROMPTS  # noqa: E402

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_named_regime(name: str) -> SamplingCfg:
    """Load a regime YAML from shared/configs/regime/<name>.yaml as a SamplingCfg."""
    path = _REPO_ROOT / "shared" / "configs" / "regime" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown regime '{name}' (looked for {path}).")
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(SamplingCfg)
    merged = OmegaConf.merge(schema, raw)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def _resolve_entry(
    entry: PromptRunEntryCfg, default_regime: SamplingCfg, default_max_tokens: int
) -> tuple[Prompt, SamplingCfg, int]:
    if entry.prompt_id not in PROMPTS:
        raise KeyError(
            f"Unknown prompt_id '{entry.prompt_id}'. Known: {sorted(PROMPTS.keys())}"
        )
    prompt = PROMPTS[entry.prompt_id]
    sampling = _load_named_regime(entry.regime) if entry.regime else deepcopy(default_regime)
    max_tokens = entry.max_tokens if entry.max_tokens is not None else default_max_tokens
    return prompt, sampling, max_tokens


def _make_run_id(prompt_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prompt_id}__{ts}"


def _confirm_long_run(items: list[dict], sampling: SamplingCfg, max_tokens: int, prompt_id: str) -> None:
    # Rough estimate at ~50 tok/s for HF + bitsandbytes 4-bit.
    estimated_seconds = len(items) * sampling.n_samples * max_tokens / 50.0
    if estimated_seconds < 3600:
        return
    hours = estimated_seconds / 3600
    print(
        f"\n[long-run guard] {prompt_id}: estimated {hours:.1f}h "
        f"({len(items)} items × {sampling.n_samples} samples × {max_tokens} max_tokens). "
        f"Rerun with a smaller slice if that's too long."
    )


# ─── Per-kind dispatch ───────────────────────────────────────────────────────


def _run_single(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    """Returns per-item dicts: {response, all_responses?, n_iters, n_response_tokens}."""
    with Timer("build_chat_messages"):
        chat_messages = [build_chat_messages(item, prompt) for item in items]
    outputs = handle.generate_batch(chat_messages, sampling, max_tokens)
    rows: list[dict] = []
    for out in outputs:
        rows.append(
            {
                "response": out.responses[0],
                "all_responses": None,
                "n_iters": 1,
                "n_response_tokens": out.n_response_tokens[0] if out.n_response_tokens else 0,
            }
        )
    return rows


def _run_self_consistency(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    with Timer("build_chat_messages"):
        chat_messages = [build_chat_messages(item, prompt) for item in items]
    outputs = handle.generate_batch(chat_messages, sampling, max_tokens)
    rows: list[dict] = []
    for item, out in zip(items, outputs):
        # Score-extract every sample, then vote.
        per_sample_extracts = [score_one(item, r).extracted for r in out.responses]
        idx = vote_index(per_sample_extracts)
        rows.append(
            {
                "response": out.responses[idx],
                "all_responses": list(out.responses),
                "n_iters": 1,
                "n_response_tokens": out.n_response_tokens[idx] if out.n_response_tokens else 0,
            }
        )
    return rows


def _run_php_dispatch(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    php_results = run_php(prompt, items, handle, sampling, max_tokens, max_iters=3)
    return [
        {
            "response": r.response,
            "all_responses": r.all_responses,
            "n_iters": r.n_iters,
            "n_response_tokens": r.n_response_tokens,
        }
        for r in php_results
    ]


def _run_self_refine_dispatch(
    prompt: Prompt,
    items: list[dict],
    handle: ModelHandle,
    sampling: SamplingCfg,
    max_tokens: int,
) -> list[dict]:
    refine_results = run_self_refine(prompt, items, handle, sampling, max_tokens)
    return [
        {
            "response": r.response,
            "all_responses": r.all_responses,
            "n_iters": r.n_iters,
            "n_response_tokens": r.n_response_tokens,
        }
        for r in refine_results
    ]


_DISPATCH = {
    "single": _run_single,
    "self_consistency": _run_self_consistency,
    "php": _run_php_dispatch,
    "self_refine": _run_self_refine_dispatch,
}


# ─── Per-prompt orchestration ────────────────────────────────────────────────


def _score_and_save(
    prompt: Prompt,
    items: list[dict],
    gen_rows: list[dict],
    sampling: SamplingCfg,
    max_tokens: int,
    handle: ModelHandle,
    invocation_results_dir: Path,
) -> dict:
    run_id = _make_run_id(prompt.id)
    rows: list[ResultRow] = []
    correct_count = 0
    mcq_correct = mcq_total = free_correct = free_total = 0
    boxed_count = 0
    total_tokens = 0

    for item, gen in zip(items, gen_rows):
        scored = score_one(item, gen["response"])
        if scored.correct:
            correct_count += 1
        if scored.finished_with_box:
            boxed_count += 1
        total_tokens += gen["n_response_tokens"]
        is_mcq = bool(item.get("options"))
        if is_mcq:
            mcq_total += 1
            mcq_correct += int(scored.correct)
        else:
            free_total += 1
            free_correct += int(scored.correct)

        rows.append(
            ResultRow(
                id=int(item["id"]),
                run_id=run_id,
                prompt_id=prompt.id,
                is_mcq=is_mcq,
                gold=item["answer"],
                response=gen["response"],
                extracted=scored.extracted,
                correct=scored.correct,
                finished_with_box=scored.finished_with_box,
                n_response_tokens=int(gen["n_response_tokens"]),
                n_iters=int(gen["n_iters"]),
                sampling={
                    "temperature": sampling.temperature,
                    "top_p": sampling.top_p,
                    "top_k": sampling.top_k,
                    "n_samples": sampling.n_samples,
                },
                max_tokens=max_tokens,
                all_responses=gen.get("all_responses"),
            )
        )

    out_path = invocation_results_dir / f"{run_id}.jsonl"
    save_jsonl(rows, out_path)

    n = len(items)
    return {
        "prompt_id": prompt.id,
        "run_id": run_id,
        "n": n,
        "overall_acc": correct_count / n if n else 0.0,
        "mcq_acc": mcq_correct / mcq_total if mcq_total else float("nan"),
        "free_acc": free_correct / free_total if free_total else float("nan"),
        "avg_tokens": total_tokens / n if n else 0,
        "pct_boxed": boxed_count / n if n else 0.0,
        "regime": sampling.name,
        "max_tokens": max_tokens,
        "out_path": str(out_path),
    }


def _print_leaderboard(rows: list[dict], slice_label: str, max_tokens_default: int) -> None:
    print()
    print("=" * 110)
    print(
        f"PROMPT ENGINEERING LEADERBOARD  "
        f"(slice={slice_label}, default_max_tokens={max_tokens_default})"
    )
    print("=" * 110)
    print(
        f"{'prompt_id':<24} {'overall':>8} {'MCQ':>8} {'free':>8} "
        f"{'avg_tok':>8} {'pct_box':>8} {'regime':>16} {'max_tok':>8}"
    )
    print("-" * 110)
    for r in rows:
        mcq = "n/a" if r["mcq_acc"] != r["mcq_acc"] else f"{r['mcq_acc']*100:.1f}%"
        free = "n/a" if r["free_acc"] != r["free_acc"] else f"{r['free_acc']*100:.1f}%"
        print(
            f"{r['prompt_id']:<24} "
            f"{r['overall_acc']*100:>7.1f}% "
            f"{mcq:>8} "
            f"{free:>8} "
            f"{r['avg_tokens']:>8.0f} "
            f"{r['pct_boxed']*100:>7.1f}% "
            f"{r['regime']:>16} "
            f"{r['max_tokens']:>8}"
        )
    print("=" * 110)


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_metrics_files(
    rows: list[dict],
    output_dir: Path,
    *,
    run_name: str,
    eval_cfg: EvalSliceCfg,
    runner_cfg: RunnerCfg,
    model_id: str,
    adapter_path: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "model_id": model_id,
        "adapter_path": adapter_path or None,
        "engine": runner_cfg.engine,
        "quant": runner_cfg.quant,
        "slice": eval_cfg.slice,
        "slice_indices": eval_cfg.slice_indices,
        "data_path": eval_cfg.data_path,
        "default_max_tokens": eval_cfg.max_tokens,
        "rows": rows,
    }
    safe_payload = _json_safe(payload)
    (output_dir / "metrics.json").write_text(json.dumps(safe_payload, indent=2) + "\n")
    save_jsonl(rows, output_dir / "leaderboard.jsonl")

    csv_path = output_dir / "leaderboard.csv"
    fieldnames = [
        "prompt_id",
        "run_id",
        "n",
        "overall_acc",
        "mcq_acc",
        "free_acc",
        "avg_tokens",
        "pct_boxed",
        "regime",
        "max_tokens",
        "out_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k)) for k in fieldnames})


def _init_wandb(
    cfg: DictConfig,
    eval_cfg: EvalSliceCfg,
    runner_cfg: RunnerCfg,
    run_dir_name: str,
    adapter_path: str,
):
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        print("W&B logging disabled: WANDB_PROJECT is not set.")
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "WANDB_PROJECT is set, but wandb is not installed. "
            "Install it with `python -m pip install wandb`."
        ) from e

    tags = [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t]
    run = wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=os.environ.get("WANDB_NAME") or run_dir_name,
        group=os.environ.get("WANDB_GROUP") or None,
        tags=tags,
        config=_json_safe(
            {
                "run_name": run_dir_name,
                "model_id": MODEL_ID,
                "adapter_path": adapter_path or None,
                "eval": OmegaConf.to_container(cfg.eval, resolve=True),
                "runner": OmegaConf.to_container(cfg.runner, resolve=True),
                "run": OmegaConf.to_container(cfg.run, resolve=True),
                "engine": runner_cfg.engine,
                "quant": runner_cfg.quant,
                "slice": eval_cfg.slice,
                "slice_indices": eval_cfg.slice_indices,
                "data_path": eval_cfg.data_path,
                "default_max_tokens": eval_cfg.max_tokens,
            }
        ),
    )
    print(f"W&B logging enabled: project={project}, run={run.name}, url={run.url}")
    return run


def _log_wandb_row(wandb_run, row: dict, elapsed: float) -> None:
    if wandb_run is None:
        return
    import wandb

    prefix = row["prompt_id"]
    metrics = {
        f"{prefix}/overall_acc": row["overall_acc"],
        f"{prefix}/mcq_acc": None if row["mcq_acc"] != row["mcq_acc"] else row["mcq_acc"],
        f"{prefix}/free_acc": None if row["free_acc"] != row["free_acc"] else row["free_acc"],
        f"{prefix}/avg_tokens": row["avg_tokens"],
        f"{prefix}/pct_boxed": row["pct_boxed"],
        f"{prefix}/n": row["n"],
        f"{prefix}/max_tokens": row["max_tokens"],
        f"{prefix}/wall_seconds": elapsed,
        "overall_acc": row["overall_acc"],
        "avg_tokens": row["avg_tokens"],
        "pct_boxed": row["pct_boxed"],
        "wall_seconds": elapsed,
    }
    wandb.log(_json_safe(metrics))


def _finish_wandb(wandb_run, rows: list[dict], metrics_dirs: list[Path]) -> None:
    if wandb_run is None:
        return
    import wandb

    table = wandb.Table(
        columns=[
            "prompt_id",
            "run_id",
            "n",
            "overall_acc",
            "mcq_acc",
            "free_acc",
            "avg_tokens",
            "pct_boxed",
            "regime",
            "max_tokens",
            "out_path",
        ]
    )
    for row in rows:
        table.add_data(
            row["prompt_id"],
            row["run_id"],
            row["n"],
            row["overall_acc"],
            None if row["mcq_acc"] != row["mcq_acc"] else row["mcq_acc"],
            None if row["free_acc"] != row["free_acc"] else row["free_acc"],
            row["avg_tokens"],
            row["pct_boxed"],
            row["regime"],
            row["max_tokens"],
            row["out_path"],
        )
    wandb.log({"leaderboard": table})

    artifact = wandb.Artifact(f"{wandb_run.name}-metrics", type="eval-metrics")
    for metrics_dir in metrics_dirs:
        for name in ("metrics.json", "leaderboard.csv", "leaderboard.jsonl"):
            path = metrics_dir / name
            if path.exists():
                artifact.add_file(str(path), name=f"{metrics_dir.name}/{name}")
    wandb_run.log_artifact(artifact)
    print(f"W&B metrics artifact logged: {artifact.name}")
    wandb_run.finish()
    print("W&B run finished.")


# ─── Hydra entrypoint ────────────────────────────────────────────────────────


register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Convert the Hydra DictConfig into typed dataclass instances. We merge with
    # the structured schema first so OmegaConf.to_object yields the dataclass
    # type (not a plain dict).
    def _typed(node, schema):
        merged = OmegaConf.merge(OmegaConf.structured(schema), node)
        return OmegaConf.to_object(merged)

    if cfg.get("prompt_id") is not None:
        OmegaConf.update(cfg, "run.runs", [{"prompt_id": cfg.prompt_id}], merge=False)

    eval_cfg: EvalSliceCfg = _typed(cfg.eval, EvalSliceCfg)  # type: ignore[assignment]
    default_regime: SamplingCfg = _typed(cfg.regime, SamplingCfg)  # type: ignore[assignment]
    runner_cfg: RunnerCfg = _typed(cfg.runner, RunnerCfg)  # type: ignore[assignment]
    adapter_path = resolve_adapter_path(runner_cfg)
    runs: list[PromptRunEntryCfg] = [_typed(e, PromptRunEntryCfg) for e in cfg.run.runs]

    if cfg.get("run_name"):
        run_dir_name = str(cfg.run_name)
    else:
        invocation_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_group = HydraConfig.get().runtime.choices["run"]
        run_dir_name = f"{run_group}_{invocation_ts}"
    invocation_results_dir = Path(cfg.results_dir) / run_dir_name
    extra_metrics_dir = (
        Path(os.environ["EVAL_METRICS_DIR"]) / run_dir_name
        if os.environ.get("EVAL_METRICS_DIR")
        else None
    )

    items = load_eval_slice(eval_cfg)
    print(
        f"Loaded {len(items)} items from {eval_cfg.data_path} "
        f"(slice={eval_cfg.slice}, max_tokens={eval_cfg.max_tokens})."
    )
    print(f"Writing results to {invocation_results_dir}")
    if extra_metrics_dir:
        print(f"Mirroring metrics to {extra_metrics_dir}")
    if not items:
        print("Empty eval slice; nothing to do.")
        return

    # Cost guard: if any run looks long, warn early
    for entry in runs:
        try:
            prompt, sampling, max_tokens = _resolve_entry(entry, default_regime, eval_cfg.max_tokens)
        except Exception:
            continue
        _confirm_long_run(items, sampling, max_tokens, prompt.id)

    run_registry = TimingsRegistry()
    wandb_run = _init_wandb(cfg, eval_cfg, runner_cfg, run_dir_name, adapter_path)

    adapter_msg = f", adapter={adapter_path}" if adapter_path else ""
    print(
        f"Loading model: {MODEL_ID} "
        f"(engine={runner_cfg.engine}, quant={runner_cfg.quant}{adapter_msg})"
    )
    with use_registry(run_registry):
        handle = load_model(runner_cfg)
    print("Model loaded.")

    leaderboard_rows: list[dict] = []
    for entry in runs:
        prompt, sampling, max_tokens = _resolve_entry(entry, default_regime, eval_cfg.max_tokens)
        if prompt.kind not in _DISPATCH:
            raise ValueError(f"Unknown prompt.kind '{prompt.kind}' for prompt {prompt.id}.")
        runner_fn = _DISPATCH[prompt.kind]
        print(
            f"\n→ {prompt.id} ({prompt.kind}, regime={sampling.name}, "
            f"max_tokens={max_tokens}, n_samples={sampling.n_samples})"
        )
        prompt_registry = TimingsRegistry()
        t0 = time.time()
        with use_registry(prompt_registry):
            with Timer("generate.dispatch"):
                gen_rows = runner_fn(prompt, items, handle, sampling, max_tokens)
            with Timer("score_and_save"):
                leaderboard = _score_and_save(
                    prompt, items, gen_rows, sampling, max_tokens, handle, invocation_results_dir
                )
        elapsed = time.time() - t0
        leaderboard_rows.append(leaderboard)
        print(
            f"   acc={leaderboard['overall_acc']*100:.1f}% "
            f"(MCQ={leaderboard['mcq_acc']*100:.1f}% free={leaderboard['free_acc']*100:.1f}%) "
            f"pct_boxed={leaderboard['pct_boxed']*100:.1f}% "
            f"avg_tok={leaderboard['avg_tokens']:.0f} "
            f"wall={elapsed:.1f}s "
            f"→ {leaderboard['out_path']}"
        )
        print(prompt_registry.render_table(f"timings — {prompt.id}"))
        run_registry.merge(prompt_registry)
        _log_wandb_row(wandb_run, leaderboard, elapsed)

        metrics_dirs = [invocation_results_dir]
        if extra_metrics_dir:
            metrics_dirs.append(extra_metrics_dir)
        for metrics_dir in metrics_dirs:
            _write_metrics_files(
                leaderboard_rows,
                metrics_dir,
                run_name=run_dir_name,
                eval_cfg=eval_cfg,
                runner_cfg=runner_cfg,
                model_id=MODEL_ID,
                adapter_path=adapter_path,
            )

    _print_leaderboard(leaderboard_rows, eval_cfg.slice, eval_cfg.max_tokens)
    print(run_registry.render_table("timings — full run"))
    metrics_dirs = [invocation_results_dir]
    if extra_metrics_dir:
        metrics_dirs.append(extra_metrics_dir)
    for metrics_dir in metrics_dirs:
        _write_metrics_files(
            leaderboard_rows,
            metrics_dir,
            run_name=run_dir_name,
            eval_cfg=eval_cfg,
            runner_cfg=runner_cfg,
            model_id=MODEL_ID,
            adapter_path=adapter_path,
        )
    _finish_wandb(wandb_run, leaderboard_rows, metrics_dirs)


if __name__ == "__main__":
    main()
