# GRPO RL — writeup

Living document. Update with every meaningful run.

## 1. Hypothesis / thought process

Static SFT teaches format and brevity but plateaus on **correctness** because the
target is a fixed reference string. GRPO optimizes the metric we're actually scored
on: it samples a *group* of completions per prompt, scores each with the answer
checker, and pushes the policy toward the above-average ones (group-relative
advantage — no learned critic, cheaper than PPO).

The key design choice: **the reward is the eval scorer.** Both call
`shared/scoring.py::score_one`, so reward `1.0` ⇔ a completion that would be marked
correct at eval. There is no proxy objective to drift from.

- **Reward:** `correctness(0/1) + shaping`, in `src/reward.py`. Correctness is
  `score_one(...).correct` (the eval scorer). Shaping is a small, bounded term —
  a brevity bonus on *correct* answers (decaying across the completion budget) plus
  a format bonus / truncation penalty on *wrong* ones. Shaping was added after
  `grpo_demo_v1` (see §3): correctness-only at `G=2` left most groups all-correct →
  zero advantage → no learning. Run `python -m experiments.reinforcement_learning.src.reward`
  for the reward distribution + group-signal evaluation.
- **Init:** base `Qwen/Qwen3-4B-Thinking-2507` by default; `model.init_adapter_path`
  warm-starts from an SFT adapter once one exists (the README's recommended path).
- **No leakage:** train on `public.jsonl[100:]`, evaluate on `slice=default`
  (`[0:100]`). Disjoint. **RL eval must use `default`, not `full`** — `full` overlaps
  the training set and would report an inflated number.

## 2. Setup / reproduce

Install the RL extra (adds TRL; compatible with the pinned transformers 4.x / vllm 0.10):

```bash
pip install -e ".[rl]"        # or: pip install "trl>=0.17,<0.20"
```

Everything runs from the repo root with `REPO_ROOT=$PWD` (also makes the Hydra
searchpath resolve on Colab where cwd is `/content`). Target runtime: Colab **A100**.

```bash
# 0. plumbing smoke (2 steps, HF rollout, 8 rows) — just checks the loop + save path
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.train_grpo grpo=smoke run_name=grpo_smoke

# 1. baseline eval (base model, no adapter)
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.eval run_name=grpo_base_eval

# 2. demo train (50 steps, vLLM colocate; auto-falls back to HF if vLLM unavailable)
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.train_grpo grpo=demo run_name=grpo_demo_v1

# 3. eval the trained adapter on the first-100 slice
REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.eval \
  adapter_path=experiments/reinforcement_learning/results/grpo_demo_v1/final_adapter \
  run_name=grpo_demo_v1_eval
```

Config groups: `grpo/{demo,smoke}` (rollout + optimization), `model/default`
(init checkpoint), `lora/r16a32` (adapter, matches SFT). Override anything on the
CLI, e.g. `grpo.use_vllm=false grpo.num_generations=8 grpo.max_steps=200`.

Adapters land in `results/<run_name>/final_adapter/`; eval JSONL (canonical
`ResultRow` schema) in `results/<run_name>.jsonl`.

## 3. Current progress / baseline numbers

_Eval slice: first 100 of `data/public.jsonl`, greedy._

| Run | Init | Steps | Overall | MCQ | Free-form |
|---|---|---|---|---|---|
| baseline (base model) | — | — | _TBD_ | _TBD_ | _TBD_ |
| grpo_demo_v1 | base | 50 | _no real learning — see below_ | | |
| grpo_demo_v2 (G=4 + shaping) | base | 50 | _TBD_ | _TBD_ | _TBD_ |

(Fill in after running steps 1–3 above on the A100.)

### grpo_demo_v1 post-mortem (the run that didn't learn)

The 50-step run completed but the policy barely moved. Two coupled causes, both
visible in the `log_completions` reward/advantage table:

1. **Zero-advantage collapse.** `num_generations=2` is the statistical worst case:
   a 2-member group only carries gradient when it splits correct/incorrect. The base
   model already solves most of these problems, so the dominant group outcome was
   `{1,1}` → `std=0` → advantage `0.00` → no update. A large fraction of the logged
   groups show advantage exactly `0.00`.
2. **Truncation = false negatives.** Qwen3-Thinking rambled (`"Wait, wait, wait…"`
   loops) into the 4096-token cap and got cut off with no `\boxed{}` → scored `0`,
   indistinguishable from a confident wrong answer. (Teacher reference traces are
   ~130 tokens median; rollouts were 10–30× longer.)

**Fixes shipped:** (A) `G=2→4`, `completion 4096→3072`, `bs 2→4`, `ga 4→2`,
`vllm_util 0.3→0.25`. (B) reward shaping in `src/reward.py`. The shaping is the
bigger lever — its brevity term gives even all-correct groups non-zero variance.
The `reward.py` evaluation harness, run over the 1126 real teacher traces + synthetic
rollout failure modes, quantifies it: at ~85% model accuracy the fraction of groups
with non-zero advantage goes **25% → 98% at G=2**, and **48% → 100% at G=4**.

## 4. Open questions / next steps

- **Run the baseline + `grpo_demo_v2` on A100** and populate the table — confirm the
  G=4 + shaping changes turn into real eval movement (the whole point of the rewrite).
- **Watch the new diagnostics in `log_completions`:** `reward/correctness` (the honest
  accuracy signal, undistorted by shaping) and `reward/shaping` should both be logged.
  Healthy = `correctness` trending up and mean completion length trending *down*. If
  `shaping` dominates `correctness`, lower `reward_brevity_weight`.
- **Memory / completion length:** `max_completion_length` is now **8192** with
  `use_liger_loss=true` (adds `liger-kernel` to the `rl` extra). Liger chunks the GRPO
  loss over the sequence dim, so the `[B, prompt+completion, vocab]` fp32 logits (~30 GiB
  at the old 3072) is never materialized — peak logits mem ≈ `[B, chunk, vocab]` ≈ few GiB,
  ~independent of length. So 8k peak should sit *under* the old 3072 peak; the early runs
  truncated before `\boxed{}` (all-truncated groups → zero advantage), which 8k fixes.
  `train_grpo` fails fast if `use_liger_loss=true` and liger-kernel is missing. If 8k still
  OOMs: lower `max_completion_length`, then fall back to the deferred levers (`beta=0` to
  drop the reference forward; vLLM sleep/offload). smoke.yaml keeps `use_liger_loss=false`.
- ~~**Reward sparsity** / zero advantage on all-right groups~~ — addressed by the
  shaping reward (see §3). Revisit `num_generations` higher (8) if signal is still thin.
- **MCQ brevity-hack watch:** brevity is gated on *correct*, so short-wrong gains
  nothing, but on MCQ a correct 1-token `\boxed{F}` guess still earns full brevity.
  The math (gating + `weight 0.3 < 1`) makes guess-then-stop unprofitable vs.
  reason-then-answer, but verify MCQ reasoning length doesn't crater via `log_completions`.
- **Reward-hacking watch:** `score_one`'s MCQ path accepts a bare trailing capital
  letter even without `\boxed{}`. Watch MCQ vs free-form reward separately; tighten
  `beta` if the policy drifts to degenerate outputs.
- **Warm-start from SFT** (`model.init_adapter_path=<sft adapter>`) once the SFT track
  has a checkpoint that beats the prompt baseline — the README's gating condition.
