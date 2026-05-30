"""Reward functions for GRPO, plus a GPU-free evaluation harness.

Two problems showed up in the first demo run (`grpo_demo_v1`):

  1. **Zero-advantage collapse.** With a binary 0/1 correctness reward and a
     small group (`num_generations=2`), most groups come back all-correct
     (`{1,1}`) because Qwen3-4B-Thinking already solves these grade-school
     problems. GRPO's advantage is `(r - mean)/std`; an all-equal group has
     `std=0` → advantage 0 → no gradient. The bulk of the run learned nothing.

  2. **Truncation = false negatives.** The model emits enormous `<think>`
     traces (the teacher references median ~130 tokens; rollouts ran into the
     4096-token cap and got cut off mid-sentence with no `\\boxed{}`). A
     truncated trace scores 0 — identical to a confident wrong answer — so the
     rare learning signal is polluted, and nothing pushes back on the rambling.

The reward here is **additive and correctness-dominant**, exposed as two
functions so TRL logs each component separately:

  * `correctness`  — pure 1.0/0.0 from the eval scorer (the real objective).
  * `shaping`      — a small, bounded term that (a) gives *correct* answers a
                     brevity bonus that decays across the whole completion
                     budget, and (b) for *wrong* answers, rewards committing to
                     a parseable `\\boxed{}` and penalizes runaway truncation.

The brevity bonus is the load-bearing fix for problem (1): two correct rollouts
of *different length* now get different rewards, so the dominant `{1,1}` groups
have non-zero variance and produce gradient again — pushing the policy toward
*short, correct, terminating* traces, which is also what cures (2).

Total reward per sample (with default weights):

    correct & short   →  1.0 + 0.3·brevity   ∈ (1.0, 1.3]
    correct & long    →  ~1.0
    wrong but boxed   →  0.0 + 0.10           (committed to an answer)
    wrong, no box     →  0.0
    truncated runaway →  0.0 - 0.10           (worst)

Correctness always dominates (`brevity_weight < 1`): a correct-but-long answer
(1.0) still beats a wrong-but-short one (≤0.1), so brevity never trades away
accuracy. All shaping terms are bounded, so the reward can't be farmed.

Run the evaluation harness (no GPU / no TRL needed):
    REPO_ROOT=$PWD python -m experiments.reinforcement_learning.src.reward
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("REPO_ROOT", str(_REPO_ROOT))

from shared.scoring import score_one  # noqa: E402


# ─── Defaults (overridable via build_reward_funcs / config) ──────────────────

DEFAULT_BREVITY_WEIGHT = 0.3   # max brevity bonus added to a *correct* answer
DEFAULT_FORMAT_WEIGHT = 0.10   # bonus for a *wrong* answer that still emits a box
DEFAULT_TRUNC_WEIGHT = 0.10    # penalty for a wrong answer that ran off the token cap
_CHARS_PER_TOKEN = 4           # length proxy when token ids aren't available


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _completion_text(completion) -> str:
    """Pull assistant text out of a completion (conversational → list of messages)."""
    if isinstance(completion, list):
        if not completion:
            return ""
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return completion or ""


def _length_tokens(text: str, ids) -> int:
    """Completion length in tokens — exact from `completion_ids`, else a char proxy."""
    if ids is not None:
        try:
            return len(ids)
        except TypeError:
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _is_truncated(text: str, ids, max_completion_length: int | None) -> bool:
    """Did generation hit the cap without finishing?

    Exact when token ids are available (length reached the cap). Otherwise a
    text proxy: a thinking model that neither closed `</think>` nor emitted a
    `\\boxed{}` almost certainly got cut off mid-trace.
    """
    if ids is not None and max_completion_length:
        try:
            if len(ids) >= max_completion_length:
                return True
        except TypeError:
            pass
    return ("</think>" not in text) and ("\\boxed" not in text)


def _score(item: dict, text: str):
    try:
        return score_one(item, text)
    except Exception:  # noqa: BLE001 — a raised reward aborts the whole training step
        from shared.scoring import ScoredResult

        return ScoredResult(correct=False, extracted="", finished_with_box=False)


def _item_from(ans_j: str, opt_j: str) -> dict:
    import json

    item = {"answer": json.loads(ans_j)}
    options = json.loads(opt_j)
    if options:  # truthy → MCQ; None/[] → free-form (mirrors score_one's dispatch)
        item["options"] = options
    return item


# ─── Core scalar reward (single sample; unit-testable) ───────────────────────


def correctness_value(item: dict, text: str) -> float:
    """1.0 if the boxed answer is correct, else 0.0."""
    return 1.0 if _score(item, text).correct else 0.0


def shaping_value(
    item: dict,
    text: str,
    length_tokens: int,
    truncated: bool,
    *,
    brevity_weight: float = DEFAULT_BREVITY_WEIGHT,
    brevity_soft_len: int = 4096,
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
    trunc_weight: float = DEFAULT_TRUNC_WEIGHT,
) -> float:
    """Bounded shaping term added on top of correctness.

    `brevity_soft_len` is the length at which the brevity bonus decays to 0 —
    set it to the completion budget so the bonus stays informative across the
    model's *actual* (very long) length distribution, not just near zero.
    """
    scored = _score(item, text)
    if scored.correct:
        brevity = max(0.0, 1.0 - length_tokens / max(1, brevity_soft_len))
        return brevity_weight * brevity
    r = 0.0
    if scored.finished_with_box:
        r += format_weight
    if truncated:
        r -= trunc_weight
    return r


# ─── TRL reward-function factory ─────────────────────────────────────────────


def build_reward_funcs(
    *,
    max_completion_length: int,
    brevity_weight: float = DEFAULT_BREVITY_WEIGHT,
    brevity_soft_len: int | None = None,
    format_weight: float = DEFAULT_FORMAT_WEIGHT,
    trunc_weight: float = DEFAULT_TRUNC_WEIGHT,
):
    """Return `[correctness, shaping]` callables for `GRPOTrainer(reward_funcs=...)`.

    TRL passes every non-`prompt` dataset column through as a list aligned to
    `completions`, plus `completion_ids` (when available) and `prompts` in
    `**kwargs`. We score twice (once per function) so each shows up as its own
    `reward/<name>` metric — cheap relative to generation, and the separate
    `reward/correctness` curve is the honest "is it actually getting them right"
    signal, undistorted by shaping.
    """
    soft_len = brevity_soft_len or max_completion_length

    def correctness(completions, answer_json, options_json, **kwargs) -> list[float]:
        out = []
        for completion, ans_j, opt_j in zip(completions, answer_json, options_json):
            out.append(correctness_value(_item_from(ans_j, opt_j), _completion_text(completion)))
        return out

    def shaping(completions, answer_json, options_json, **kwargs) -> list[float]:
        completion_ids = kwargs.get("completion_ids")
        out = []
        for i, (completion, ans_j, opt_j) in enumerate(zip(completions, answer_json, options_json)):
            text = _completion_text(completion)
            ids = completion_ids[i] if completion_ids is not None and i < len(completion_ids) else None
            length = _length_tokens(text, ids)
            truncated = _is_truncated(text, ids, max_completion_length)
            out.append(
                shaping_value(
                    _item_from(ans_j, opt_j), text, length, truncated,
                    brevity_weight=brevity_weight, brevity_soft_len=soft_len,
                    format_weight=format_weight, trunc_weight=trunc_weight,
                )
            )
        return out

    correctness.__name__ = "correctness"
    shaping.__name__ = "shaping"
    return [correctness, shaping]


# ─── Evaluation harness (run as a module) ────────────────────────────────────


def _evaluate() -> None:
    """Validate the reward on real teacher traces + synthetic rollout failures,
    and quantify how much group-relative signal the shaping restores."""
    import json
    import random
    import statistics as st

    rows = [json.loads(l) for l in open(_REPO_ROOT / "data/public.jsonl")]
    soft_len = 3072  # = the proposed max_completion_length in demo.yaml

    def total(item, text, length, truncated):
        return correctness_value(item, text) + shaping_value(
            item, text, length, truncated, brevity_soft_len=soft_len
        )

    # (1) Real teacher traces: correct, concise, terminate with a box.
    print("── (1) Reward on the 1126 real teacher traces (correct & concise) ──")
    tvals, tlens = [], []
    for r in rows:
        text = r["llm_response"]
        L = max(1, len(text) // _CHARS_PER_TOKEN)
        item = {"answer": r["answer"]}
        if r.get("options"):
            item["options"] = r["options"]
        tvals.append(total(item, text, L, truncated=False))
        tlens.append(L)
    print(f"   length(tok)  median={int(st.median(tlens))}  p90={sorted(tlens)[int(.9*len(tlens))]}  max={max(tlens)}")
    print(f"   reward       min={min(tvals):.3f}  median={st.median(tvals):.3f}  max={max(tvals):.3f}")
    print(f"   → all correct, brevity spreads them over ({min(tvals):.2f}, {max(tvals):.2f}] "
          f"instead of a flat 1.0\n")

    # (2) Synthetic rollout failure modes, mirroring the observed completions.
    print("── (2) Reward ordering across observed rollout shapes (one free-form item) ──")
    item = {"answer": ["8"]}                       # the "288 moths → 8" problem
    box_ok = "Parse: ... Compute: 48/6=8.\nAnswer: \\boxed{8}"
    box_bad = "Parse: ... Compute: 48/6=9.\nAnswer: \\boxed{9}"
    ramble = "<think>\n" + "Wait, let me re-check. " * 600          # ~3000 tok, no close
    cases = [
        ("correct, concise (~120 tok)",  box_ok,                       120,  False),
        ("correct, verbose (~2500 tok)", box_ok + " filler" * 0,       2500, False),
        ("wrong, boxed (~150 tok)",      box_bad,                      150,  False),
        ("wrong, no box (~200 tok)",     "Parse: hmm I am unsure.",    200,  False),
        ("truncated runaway (3072 tok)", ramble,                       3072, True),
    ]
    for name, text, L, trunc in cases:
        c = correctness_value(item, text)
        s = shaping_value(item, text, L, trunc, brevity_soft_len=soft_len)
        print(f"   {name:32s}  correctness={c:.1f}  shaping={s:+.3f}  total={c+s:+.3f}")
    print()

    # (3) The central fix: fraction of GRPO groups with non-zero advantage.
    #     Model each rollout as correct w.p. p (the model's accuracy) with a
    #     length drawn from a heavy, partly-truncating distribution like the one
    #     observed. A group yields gradient iff its rewards aren't all equal.
    print("── (3) Fraction of groups with NON-ZERO advantage (std>0 → gradient) ──")
    rng = random.Random(0)
    N = 20000

    def sample_len():  # lognormal-ish: lots of long traces, some hit the 3072 cap
        L = int(rng.lognormvariate(7.0, 0.7))      # median ~1100 tok, long right tail
        return min(L, soft_len), (L >= soft_len)

    def group_has_signal(G, p, reward_fn):
        rs = []
        for _ in range(G):
            correct = rng.random() < p
            L, trunc = sample_len()
            rs.append(reward_fn(correct, L, trunc))
        return max(rs) - min(rs) > 1e-9

    old_reward = lambda correct, L, trunc: 1.0 if correct else 0.0          # noqa: E731
    new_reward = lambda correct, L, trunc: (                                  # noqa: E731
        1.0 + DEFAULT_BREVITY_WEIGHT * max(0.0, 1.0 - L / soft_len) if correct
        else (-DEFAULT_TRUNC_WEIGHT if trunc else 0.0)
    )

    print(f"   {'setup':22s} {'OLD (binary)':>14s} {'NEW (shaped)':>14s}")
    for p in (0.85, 0.70, 0.50):
        for G in (2, 4):
            old = st.mean(group_has_signal(G, p, old_reward) for _ in range(N))
            new = st.mean(group_has_signal(G, p, new_reward) for _ in range(N))
            print(f"   acc={p:.2f}  G={G:<2d}        {old*100:11.1f} %  {new*100:11.1f} %")
    print("\n   OLD: signal only when a group happens to split correct/incorrect.")
    print("   NEW: brevity makes near-every group informative — even all-correct ones.")


if __name__ == "__main__":
    _evaluate()
