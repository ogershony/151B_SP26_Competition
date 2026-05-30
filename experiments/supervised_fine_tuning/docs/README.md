# Supervised Fine-Tuning (SFT)

**Goal:** Bias the model toward (a) committing to a `\boxed{}` answer earlier, (b) following the MCQ-letter format reliably, and (c) producing shorter, more correct chains-of-thought.

**Approach:**
- LoRA / QLoRA on top of Qwen3-4B-Thinking-2507 — full fine-tuning is overkill for the compute we have.
- Training data: math-reasoning datasets with short, formatted solutions (e.g. GSM8K-style, MATH, plus self-generated traces filtered by the judger).
- Likely use TRL's `SFTTrainer` or similar.

**Why:** Prompt engineering can only push the model so far if the base policy is verbose-by-default. SFT changes the prior — directly targets the "thinks past the token budget" failure mode visible in the starter baseline.

**Gating:** Don't start until prompt-engineering and parameter-sampling experiments have a clean, pinned baseline to compare against — otherwise it's impossible to attribute lift to SFT vs. better prompts.

**Success looks like:** Beats the best prompt-engineering + parameter-sampling baseline on the eval slice, with the largest gains on free-form (where formatting and brevity matter most).

## Command References

- `run_commands.md`: full experiment history and longer command notes.
- `mixed_27k_quick_commands.md`: concise commands for mixed 27k training,
  checkpoint resume, eval, and Google Drive upload with rclone.
