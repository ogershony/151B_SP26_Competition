"""Prompt catalog for the prompt-engineering experiment.

10 non-baseline prompts defined here; the canonical baseline is imported from
`shared.prompts.BASELINE_STARTER`. The `PROMPTS` registry at the bottom is the
single source of truth `eval.py` reads from.
"""

from shared.prompts import BASELINE_STARTER, FewShotExemplar, Prompt


# ─── Chain-of-thought (2) ────────────────────────────────────────────────────

# Explicit step-per-line CoT with an early-commit instruction.
COT_EXPLICIT = Prompt(
    id="cot_explicit",
    kind="single",
    system_free=(
        "You are an expert mathematician. Think step by step, putting each step on its "
        "own line. Once you have the answer, immediately commit to it inside \\boxed{}. "
        "Do not write extended verification after the box; if you must verify, do it in "
        "the steps before the box. "
        "If the problem has multiple sub-answers, separate them by commas inside a single "
        "\\boxed{}, e.g. \\boxed{3, 7}."
    ),
    system_mcq=(
        "You are an expert mathematician. Think step by step, one step per line, then "
        "commit to the chosen option letter inside \\boxed{}, e.g. \\boxed{C}. Output "
        "ONLY the letter."
    ),
)

# Forces a fixed structure; intended to bound the Compute stage so the model doesn't ramble.
COT_STRUCTURED = Prompt(
    id="cot_structured",
    kind="single",
    system_free=(
        "You are an expert mathematician. Solve in five named sections, each kept short:\n"
        "Parse: restate what is asked in one sentence.\n"
        "Plan: one-line strategy.\n"
        "Compute: do the math (this is the only place where work is shown; keep it terse).\n"
        "Verify: one sanity check.\n"
        "Answer: put ALL final answers inside ONE \\boxed{}. "
        "Single answer: \\boxed{42}. "
        "Multiple sub-answers, comma-separated: \\boxed{3, 7}."
    ),
    system_mcq=(
        "You are an expert mathematician. Solve in five short sections — Parse, Plan, "
        "Compute, Verify, Answer — and on the Answer line write ONLY the letter inside "
        "\\boxed{}, e.g. Answer: \\boxed{C}."
    ),
)

# ─── Few-shot (2) ────────────────────────────────────────────────────────────

_FS_FREE_BASIC = [
    FewShotExemplar(
        question="Find the sum of the first 50 positive even integers.",
        assistant_response=(
            "Sum = 2 + 4 + ... + 100 = 2 * (1 + 2 + ... + 50) = 2 * (50*51/2) = 50*51 = 2550. "
            "\\boxed{2550}"
        ),
    ),
    FewShotExemplar(
        question="Compute $\\int_0^1 (3x^2 + 2x) dx$.",
        assistant_response=(
            "Antiderivative: x^3 + x^2. Evaluate at 1: 1 + 1 = 2. At 0: 0. Difference: 2. "
            "\\boxed{2}"
        ),
    ),
]

_FS_MCQ_BASIC = [
    FewShotExemplar(
        question="What is 7 * 8?",
        options=["54", "56", "63", "64", "72"],
        assistant_response="7*8 = 56. That matches B. \\boxed{B}",
    ),
    FewShotExemplar(
        question="Which is a prime number?",
        options=["1", "9", "15", "21", "23", "25", "27", "33", "39"],
        assistant_response="23 has no divisors other than 1 and itself. That is option E. \\boxed{E}",
    ),
]

# 2 worked exemplars per modality. Terse reasoning + boxed answer style.
FEWSHOT_2_BASIC = Prompt(
    id="fewshot_2_basic",
    kind="single",
    system_free=(
        "You are an expert mathematician. Solve the problem step-by-step, briefly, and "
        "put your final answer inside \\boxed{}. Multiple sub-answers go comma-separated "
        "inside one \\boxed{}."
    ),
    system_mcq=(
        "You are an expert mathematician. Read the problem and the answer choices, then "
        "select the single best answer. Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}."
    ),
    few_shot_free=_FS_FREE_BASIC,
    few_shot_mcq=_FS_MCQ_BASIC,
)

_FS_FREE_DIVERSE = _FS_FREE_BASIC + [
    FewShotExemplar(
        question="How many ways can the letters of 'AABC' be arranged?",
        assistant_response=(
            "Total = 4! / 2! = 24 / 2 = 12 arrangements. \\boxed{12}"
        ),
    ),
]

_FS_MCQ_DIVERSE = _FS_MCQ_BASIC + [
    FewShotExemplar(
        question="If $f(x) = x^2 + 1$, what is $f(3)$?",
        options=["7", "8", "9", "10", "11"],
        assistant_response="f(3) = 9 + 1 = 10. That is option D. \\boxed{D}",
    ),
]

# 3 exemplars per modality spanning algebra, calculus, combinatorics.
FEWSHOT_3_DIVERSE = Prompt(
    id="fewshot_3_diverse",
    kind="single",
    system_free=FEWSHOT_2_BASIC.system_free,
    system_mcq=FEWSHOT_2_BASIC.system_mcq,
    few_shot_free=_FS_FREE_DIVERSE,
    few_shot_mcq=_FS_MCQ_DIVERSE,
)

# ─── Self-consistency (2) ────────────────────────────────────────────────────

# Designed to pair with high-temperature N-sample sampling. Voting handled by shared.voting.
SC_DIVERSE_PATHS = Prompt(
    id="sc_diverse_paths",
    kind="self_consistency",
    system_free=(
        "You are an expert mathematician. Solve the problem using a clear approach. "
        "Different attempts may explore different valid approaches; pick one and execute "
        "it cleanly. Put your final answer inside \\boxed{}. Multiple sub-answers go "
        "comma-separated in a single \\boxed{}."
    ),
    system_mcq=(
        "You are an expert mathematician. Pick the best answer choice using a clear approach. "
        "Different attempts may explore different valid lines of reasoning; pick one and "
        "execute it. Output ONLY the letter inside \\boxed{}."
    ),
)

# Short responses keep N-sample voting cheap.
SC_TERSE = Prompt(
    id="sc_terse",
    kind="self_consistency",
    system_free=(
        "You are an expert mathematician. Be terse — at most 3 short lines of work — then "
        "commit your answer inside \\boxed{}. Multiple sub-answers go comma-separated."
    ),
    system_mcq=(
        "You are an expert mathematician. Be terse — at most 2 short lines of work — then "
        "commit the answer letter inside \\boxed{}, e.g. \\boxed{C}."
    ),
)

# ─── Progressive-Hint Prompting (2) ──────────────────────────────────────────

# Plain PHP. Iteration handled by shared.multi_turn.run_php.
PHP_BASIC = Prompt(
    id="php_basic",
    kind="php",
    system_free=(
        "You are an expert mathematician. Solve the problem step-by-step and put the final "
        "answer inside \\boxed{}. Multiple sub-answers go comma-separated."
    ),
    system_mcq=(
        "You are an expert mathematician. Pick the best answer letter and put it inside "
        "\\boxed{}, e.g. \\boxed{C}."
    ),
    php_hint_template=(
        "Hint: a previous attempt produced \\boxed{{{prev}}}. "
        "Verify carefully and output your final answer in \\boxed{{}}."
    ),
)

# Skepticism cue to fight answer-anchoring bias.
PHP_SKEPTICAL = Prompt(
    id="php_skeptical",
    kind="php",
    system_free=PHP_BASIC.system_free,
    system_mcq=PHP_BASIC.system_mcq,
    php_hint_template=(
        "Hint: a previous attempt produced \\boxed{{{prev}}}. Treat this as a candidate, "
        "not a fact — re-derive the answer from scratch and either confirm it or correct "
        "it inside \\boxed{{}}."
    ),
)

# ─── Other inference-time techniques (3) ─────────────────────────────────────

# Targets the documented failure: thinking model burning 32k tokens before reaching
# \\boxed{}. Pairs naturally with a tighter MAX_TOKENS (e.g. 2048).
FORCE_EARLY_BOXED = Prompt(
    id="force_early_boxed",
    kind="single",
    system_free=(
        "You are an expert mathematician. The MOMENT you have a candidate final answer, "
        "write it inside \\boxed{} immediately — even if you want to verify afterward. "
        "After committing the box you may continue verifying or refining; if you change "
        "your mind, write a NEW \\boxed{} with the corrected answer (the last one wins). "
        "Multiple sub-answers go comma-separated inside one \\boxed{}, e.g. \\boxed{3, 7}.\n"
        "Strict rule: \\boxed{...} must appear within the first ~500 tokens of your reply, "
        "even if rough — do not exhaust your reasoning budget without committing."
    ),
    system_mcq=(
        "You are an expert mathematician. The MOMENT you have a best-guess letter, write "
        "\\boxed{X} immediately, then continue verifying. If you change your mind, write a "
        "new \\boxed{Y} — the last one wins. Output a candidate \\boxed{} within the first "
        "~200 tokens."
    ),
)

# Wang et al. 2023 Plan-and-Solve baseline.
PLAN_AND_SOLVE = Prompt(
    id="plan_and_solve",
    kind="single",
    system_free=(
        "You are an expert mathematician. First write a 3–5 bullet plan describing the "
        "steps you will take. Then execute the plan. End with the final answer inside "
        "\\boxed{}. Multiple sub-answers go comma-separated inside one \\boxed{}."
    ),
    system_mcq=(
        "You are an expert mathematician. First write a 3-bullet plan. Then execute. "
        "End with the chosen letter inside \\boxed{}, e.g. \\boxed{C}."
    ),
)

# Two-turn self-critique. Iteration handled by shared.multi_turn.run_self_refine.
SELF_REFINE_1PASS = Prompt(
    id="self_refine_1pass",
    kind="self_refine",
    system_free=(
        "You are an expert mathematician. Solve the problem and put the final answer "
        "inside \\boxed{}. Multiple sub-answers go comma-separated."
    ),
    system_mcq=(
        "You are an expert mathematician. Pick the best answer letter and put it inside "
        "\\boxed{}, e.g. \\boxed{C}."
    ),
    self_refine_critique=(
        "Critique your answer above for any arithmetic, algebraic, or interpretation "
        "errors. Then produce the corrected final answer inside \\boxed{}. Multiple "
        "sub-answers go comma-separated."
    ),
)


# ─── cot_structured + self-consistency ───────────────────────────────────────

# cot_structured with a diversity cue in the Plan section so majority voting
# across high_div_8 samples sees genuinely different reasoning paths.
COT_STRUCTURED_SC = Prompt(
    id="cot_structured_sc",
    kind="self_consistency",
    system_free=(
        "You are an expert mathematician. Solve in five named sections, each kept short:\n"
        "Parse: restate what is asked in one sentence.\n"
        "Plan: one-line strategy — pick a fresh approach each attempt.\n"
        "Compute: do the math (this is the only place where work is shown; keep it terse).\n"
        "Verify: one sanity check.\n"
        "Answer: put ALL final answers inside ONE \\boxed{}. "
        "Single answer: \\boxed{42}. "
        "Multiple sub-answers, comma-separated: \\boxed{3, 7}."
    ),
    system_mcq=(
        "You are an expert mathematician. Solve in five short sections — Parse, Plan, "
        "Compute, Verify, Answer — picking a fresh approach each attempt. "
        "On the Answer line write ONLY the letter inside \\boxed{}, e.g. Answer: \\boxed{C}."
    ),
)

# ─── Registry ────────────────────────────────────────────────────────────────

_LOCAL_PROMPTS = [
    COT_EXPLICIT,
    COT_STRUCTURED,
    COT_STRUCTURED_SC,
    FEWSHOT_2_BASIC,
    FEWSHOT_3_DIVERSE,
    SC_DIVERSE_PATHS,
    SC_TERSE,
    PHP_BASIC,
    PHP_SKEPTICAL,
    FORCE_EARLY_BOXED,
    PLAN_AND_SOLVE,
    SELF_REFINE_1PASS,
]

PROMPTS: dict[str, Prompt] = {BASELINE_STARTER.id: BASELINE_STARTER}
for _p in _LOCAL_PROMPTS:
    if _p.id in PROMPTS:
        raise ValueError(f"Duplicate prompt id: {_p.id}")
    PROMPTS[_p.id] = _p
