"""Hydra structured-config dataclasses, shared across all experiments.

Registering these with `ConfigStore` means YAML files in `shared/configs/` and each
experiment's `configs/` are validated against the dataclass schema at load time —
typos and type mismatches fail fast instead of deep inside a run.
"""

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EvalSliceCfg:
    slice: str = "default"
    slice_indices: Optional[list[int]] = None
    max_tokens: int = 4096
    data_path: str = "data/public.jsonl"


@dataclass
class SamplingCfg:
    name: str = "greedy"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n_samples: int = 1
    repetition_penalty: float = 1.0
    min_p: float = 0.0


@dataclass
class RunnerCfg:
    engine: str = "vllm"   # one of: "vllm", "hf"
    quant: str = "bf16"    # one of: "bf16", "bnb"
    adapter_path: Optional[str] = None


@dataclass
class PromptRunEntryCfg:
    prompt_id: str = MISSING
    regime: Optional[str] = None
    max_tokens: Optional[int] = None


@dataclass
class RunGroupCfg:
    runs: list[PromptRunEntryCfg] = field(default_factory=list)


@dataclass
class RunCfg:
    eval: EvalSliceCfg = field(default_factory=EvalSliceCfg)
    regime: SamplingCfg = field(default_factory=SamplingCfg)
    run: RunGroupCfg = field(default_factory=RunGroupCfg)
    results_dir: str = "experiments/prompt_engineering/results"
    run_name: str = "default"
    seed: int = 0


def register_configs() -> None:
    """Register dataclass schemas with Hydra's ConfigStore.

    Called from `eval.py` before `@hydra.main` is invoked.
    """
    cs = ConfigStore.instance()
    cs.store(name="run_cfg_schema", node=RunCfg)
    cs.store(group="eval", name="base_eval_cfg", node=EvalSliceCfg)
    cs.store(group="regime", name="base_regime_cfg", node=SamplingCfg)
    cs.store(group="run", name="base_run_group_cfg", node=RunGroupCfg)
    cs.store(group="runner", name="base_runner_cfg", node=RunnerCfg)
