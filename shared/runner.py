"""Model loader + batched generator.

Two engines behind a common `ModelHandle` interface:
- `_VLLMHandle` (default): vLLM continuous batching, prefix caching, single
  scheduler call per `generate_batch`.
- `_HFHandle` (fallback): Transformers + bitsandbytes-4bit, manual micro-batch
  chunking. Selected when vLLM load fails or when `engine=hf` is requested.
  We deliberately avoid `lightning.fabric.Fabric` here — for pure inference with
  `device_map="auto"`, Fabric's `launch()` spawns worker processes that grab GPU
  memory and don't release it on failure.

HF model-load code is lifted from `starter_code_cse151b_comp.ipynb` cell
`3d43b572` and generation from cell `68bad2c0`.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.schemas import RunnerCfg, SamplingCfg
from shared.telemetry import Timer

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B-Thinking-2507")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
MAX_MODEL_LEN = int(os.environ.get("RUNNER_MAX_MODEL_LEN", "16384"))
GPU_ID = "0"
MICRO_BATCH_SIZE = int(os.environ.get("RUNNER_MICRO_BATCH_SIZE", "25"))
PARALLEL_SAMPLES = _env_flag("RUNNER_PARALLEL_SAMPLES")
VLLM_USE_TQDM = _env_flag("RUNNER_VLLM_USE_TQDM", default=True)
RUNNER_VERBOSE = _env_flag("RUNNER_VERBOSE", default=True)
ADAPTER_NAME = "sft_adapter"
ADAPTER_ID = 1


def resolve_adapter_path(cfg: RunnerCfg | None = None) -> str:
    """Return the adapter path from config, falling back to ADAPTER_PATH."""
    if cfg is not None and cfg.adapter_path:
        return cfg.adapter_path
    return ADAPTER_PATH


def _validate_adapter_path(adapter_path: str) -> Path:
    path = Path(adapter_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter path does not exist: {path}")
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json under LoRA adapter path: {path}")
    if not ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()):
        raise FileNotFoundError(
            f"Missing adapter weights under LoRA adapter path: {path} "
            "(expected adapter_model.safetensors or adapter_model.bin)"
        )
    return path


def _adapter_max_rank(adapter_path: Path) -> int:
    with (adapter_path / "adapter_config.json").open() as f:
        cfg = json.load(f)

    ranks = [int(cfg.get("r", 0) or 0)]
    rank_pattern = cfg.get("rank_pattern") or {}
    if isinstance(rank_pattern, dict):
        ranks.extend(int(v) for v in rank_pattern.values() if v)
    max_rank = max(ranks)
    if max_rank <= 0:
        config_path = adapter_path / "adapter_config.json"
        raise ValueError(f"Could not determine LoRA rank from {config_path}")
    return max_rank


def _call_accepts_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


@dataclass
class GenerationOutput:
    """Per-prompt generation result (for one or more samples)."""
    responses: list[str]  # length == n_samples
    n_response_tokens: list[int]


class ModelHandle(ABC):
    """Batched-generation interface."""

    tokenizer: Any

    @abstractmethod
    def generate_batch(
        self,
        chat_messages: list[list[dict]],
        sampling: SamplingCfg,
        max_tokens: int,
    ) -> list[GenerationOutput]:
        """Generate one or more samples per prompt. Returns a list parallel to
        `chat_messages`, each entry holding `n_samples` responses.
        """
        ...


class _HFHandle(ModelHandle):
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        chat_messages: list[list[dict]],
        sampling: SamplingCfg,
        max_tokens: int,
    ) -> list[GenerationOutput]:
        import torch
        from tqdm.auto import tqdm

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = next(self.model.parameters()).device

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            repetition_penalty=sampling.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if sampling.temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = sampling.temperature
            gen_kwargs["top_p"] = sampling.top_p
            if sampling.top_k > 0:
                gen_kwargs["top_k"] = sampling.top_k

        # Loop n_samples (HF doesn't accept `n=` like vLLM); usually n=1.
        per_sample_outputs: list[list[str]] = [[] for _ in chat_messages]
        per_sample_token_counts: list[list[int]] = [[] for _ in chat_messages]

        # Chunk to bound peak GPU memory: a 100-item batch with max_new_tokens=4096
        # produces a KV cache too large for an 11 GB card. Tunable via env var.
        n_samples = max(1, sampling.n_samples)
        n_chunks = (len(chat_messages) + MICRO_BATCH_SIZE - 1) // MICRO_BATCH_SIZE
        parallel_samples = PARALLEL_SAMPLES and gen_kwargs.get("do_sample", False) and n_samples > 1
        with Timer("generate.total", cuda_sync=True):
            progress = tqdm(
                total=n_chunks if parallel_samples else n_chunks * n_samples,
                desc=f"HF generate ({len(chat_messages)} prompts x {n_samples} samples)",
                unit="sample-batch",
                dynamic_ncols=True,
            )
            for chunk_start in range(0, len(chat_messages), MICRO_BATCH_SIZE):
                chunk = chat_messages[chunk_start:chunk_start + MICRO_BATCH_SIZE]
                with Timer("generate.tokenize"):
                    prompts = [
                        self.tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True
                        )
                        for msgs in chunk
                    ]
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_MODEL_LEN,
                    ).to(device)
                    prompt_len = inputs["input_ids"].shape[1]

                if parallel_samples:
                    with Timer("generate.forward", cuda_sync=True):
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                **inputs,
                                **gen_kwargs,
                                num_return_sequences=n_samples,
                            )
                    with Timer("generate.decode"):
                        for j in range(len(chunk)):
                            for sample_idx in range(n_samples):
                                out = output_ids[j * n_samples + sample_idx]
                                new_tokens = out[prompt_len:]
                                text = self.tokenizer.decode(
                                    new_tokens, skip_special_tokens=True
                                ).strip()
                                per_sample_outputs[chunk_start + j].append(text)
                                per_sample_token_counts[chunk_start + j].append(
                                    int(new_tokens.shape[0])
                                )
                    del output_ids
                    progress.update(1)
                else:
                    for _ in range(n_samples):
                        with Timer("generate.forward", cuda_sync=True):
                            with torch.no_grad():
                                output_ids = self.model.generate(**inputs, **gen_kwargs)
                        with Timer("generate.decode"):
                            for j, out in enumerate(output_ids):
                                new_tokens = out[prompt_len:]
                                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                                per_sample_outputs[chunk_start + j].append(text)
                                per_sample_token_counts[chunk_start + j].append(int(new_tokens.shape[0]))
                        del output_ids
                        progress.update(1)

                del inputs
                torch.cuda.empty_cache()
            progress.close()

        results: list[GenerationOutput] = []
        for i in range(len(chat_messages)):
            results.append(
                GenerationOutput(
                    responses=per_sample_outputs[i],
                    n_response_tokens=per_sample_token_counts[i],
                )
            )
        return results


class _VLLMHandle(ModelHandle):
    """vLLM-backed handle. Single `llm.generate(...)` call per batch — vLLM's
    scheduler does continuous batching across items and across the `n` samples
    in `SamplingParams`, sharing prefill KV.
    """

    def __init__(self, llm: Any, lora_request: Any | None = None):
        self.llm = llm
        self.lora_request = lora_request
        self.tokenizer = llm.get_tokenizer()

    def generate_batch(
        self,
        chat_messages: list[list[dict]],
        sampling: SamplingCfg,
        max_tokens: int,
    ) -> list[GenerationOutput]:
        from vllm import SamplingParams

        with Timer("generate.tokenize"):
            prompts = [
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                for msgs in chat_messages
            ]

        sp_kwargs: dict[str, Any] = dict(
            n=max(1, sampling.n_samples),
            max_tokens=max_tokens,
            repetition_penalty=sampling.repetition_penalty,
        )
        if sampling.temperature <= 0.0:
            sp_kwargs["temperature"] = 0.0
        else:
            sp_kwargs["temperature"] = sampling.temperature
            sp_kwargs["top_p"] = sampling.top_p
            if sampling.top_k > 0:
                sp_kwargs["top_k"] = sampling.top_k
        sp = SamplingParams(**sp_kwargs)

        n_samples = max(1, sampling.n_samples)
        if RUNNER_VERBOSE:
            print(
                f"vLLM generate: {len(prompts)} prompts x {n_samples} samples "
                f"(max_tokens={max_tokens})"
            )

        generate_kwargs: dict[str, Any] = {"use_tqdm": VLLM_USE_TQDM}
        if self.lora_request is not None:
            generate_kwargs["lora_request"] = self.lora_request

        with Timer("generate.vllm"):
            outputs = self.llm.generate(prompts, sp, **generate_kwargs)

        results: list[GenerationOutput] = []
        for out in outputs:
            responses = [c.text for c in out.outputs]
            tokens = [len(c.token_ids) for c in out.outputs]
            results.append(GenerationOutput(responses=responses, n_response_tokens=tokens))
        if RUNNER_VERBOSE:
            total_sequences = sum(len(out.outputs) for out in outputs)
            total_tokens = sum(sum(len(c.token_ids) for c in out.outputs) for out in outputs)
            avg_tokens = total_tokens / total_sequences if total_sequences else 0.0
            print(
                f"vLLM generate done: {total_sequences} sequences, "
                f"{total_tokens} output tokens, avg={avg_tokens:.0f}"
            )
        return results


def _load_vllm(cfg: RunnerCfg) -> ModelHandle:
    from vllm import LLM

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_ID)

    adapter_path = resolve_adapter_path(cfg)
    adapter_dir: Path | None = None
    if adapter_path:
        adapter_dir = _validate_adapter_path(adapter_path)

    common: dict[str, Any] = dict(
        model=MODEL_ID,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        common["enable_lora"] = True
        common["max_loras"] = 1
        common["max_lora_rank"] = _adapter_max_rank(adapter_dir)

    if cfg.quant == "bf16":
        common["dtype"] = "bfloat16"
    elif cfg.quant == "bnb":
        common["quantization"] = "bitsandbytes"
        common["load_format"] = "bitsandbytes"
        if adapter_dir is not None and _call_accepts_kwarg(LLM, "qlora_adapter_name_or_path"):
            common["qlora_adapter_name_or_path"] = str(adapter_dir)
    else:
        raise ValueError(f"Unknown quant {cfg.quant!r} for vLLM; valid: 'bf16', 'bnb'")

    llm = LLM(**common)
    lora_request = None
    if adapter_dir is not None:
        from vllm.lora.request import LoRARequest

        logger.info("Loading LoRA adapter with vLLM from %s", adapter_dir)
        lora_request = LoRARequest(ADAPTER_NAME, ADAPTER_ID, str(adapter_dir))
    return _VLLMHandle(llm=llm, lora_request=lora_request)


def _load_hf(cfg: RunnerCfg) -> ModelHandle:
    if cfg.quant != "bnb":
        raise ValueError(
            f"HF path only supports quant='bnb' in this codebase (got {cfg.quant!r}). "
            "Use engine='vllm' for bf16."
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    adapter_path = resolve_adapter_path(cfg)
    if adapter_path:
        from peft import PeftModel

        adapter_dir = _validate_adapter_path(adapter_path)
        logger.info("Loading LoRA adapter from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return _HFHandle(model=model, tokenizer=tokenizer)


def load_model(cfg: RunnerCfg) -> ModelHandle:
    """Top-level dispatcher.

    `engine='vllm'` tries vLLM first; on any load-time exception falls back to
    HF + bnb-4bit (overriding the requested quant) with a warning. `engine='hf'`
    forces the HF path and validates `quant` accordingly.
    """
    with Timer("model.load"):
        if cfg.engine == "vllm":
            try:
                return _load_vllm(cfg)
            except Exception as e:
                logger.warning(
                    "vLLM load failed (%s: %s). Falling back to HF Transformers + bnb-4bit. "
                    "Requested quant=%r is being overridden to 'bnb' for the fallback path.",
                    type(e).__name__, e, cfg.quant,
                )
                fallback_cfg = RunnerCfg(
                    engine="hf",
                    quant="bnb",
                    adapter_path=resolve_adapter_path(cfg),
                )
                return _load_hf(fallback_cfg)
        elif cfg.engine == "hf":
            return _load_hf(cfg)
        else:
            raise ValueError(f"Unknown engine {cfg.engine!r}; valid: 'vllm', 'hf'")
