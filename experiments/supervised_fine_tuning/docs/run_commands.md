# SFT Run Commands

This file records QLoRA SFT training commands for `Qwen/Qwen3-4B-Thinking-2507`.
Run commands from the repo root on a GPU machine that can read and write
`/cephfs/qwen_math_comp`.

## Environment

```bash
cd /cephfs/qwen_math_comp/151B_SP26_Competition
source /cephfs/qwen_math_comp/.venv/bin/activate

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Completed Smoke Run

Known output:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_5k_smoketest/final_adapter
```

Recovered dataset/output settings from `/cephfs`:

- train: `/cephfs/qwen_math_comp/data_v2/sft_train_5k.jsonl` (5,000 rows)
- eval: `/cephfs/qwen_math_comp/data_v2/sft_val_300.jsonl` (300 rows)
- output: `/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_5k_smoketest`
- completed: 1 epoch, 313 optimizer steps
- eval loss: 0.39937 at step 313

Command:

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_v2/sft_train_5k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_v2/sft_val_300.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_numina_5k_smoketest \
  --max_seq_len 4096 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

## 10k Numina Harder Run

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_numina_10k_harder/sft_train_10k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_numina_10k_harder/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder \
  --max_seq_len 4096 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

Final adapter will be written to:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder/final_adapter
```

## 20k Numina Harder Run

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_numina_20k_harder/sft_train_20k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_numina_20k_harder/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_numina_20k_harder \
  --max_seq_len 4096 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

Final adapter will be written to:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_20k_harder/final_adapter
```

## 20k Numina 32k-Token Run

This rebuilds the SFT JSONL with the same Qwen chat-template token counting used
by training, then trains with `--max_seq_len 32768`. The long-solution character
gate is disabled with `--max_solution_chars 0`; otherwise the old 12k-character
filter can prevent genuinely long examples from entering the dataset.

```bash
python -m experiments.supervised_fine_tuning.src.prepare_numina_sft \
  --out_dir /cephfs/qwen_math_comp/data_numina_20k_32k \
  --n_train 20000 \
  --n_val 500 \
  --min_tokens 128 \
  --max_tokens 32768 \
  --max_solution_chars 0 \
  --boxed_policy last \
  --selection_strategy longest \
  --seed 42
```

Use `--selection_strategy random` only if you want the previous shuffled-first-N
behavior. For a 32k experiment, `longest` is usually the better match because it
actually fills the dataset with the longest usable rows under the cap.

Train:

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_numina_20k_32k/sft_train_20k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_numina_20k_32k/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_numina_20k_32k \
  --max_seq_len 32768 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

Final adapter will be written to:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_20k_32k/final_adapter
```

Treat this as a length-skewed ablation, not the default SFT adapter. With the
current strict Numina filters, the regenerated 20k set topped out around 4,097
tokens rather than producing true 32k-token traces.

## 27k Mixed Numina + Public-MCQ Run

This is the recommended next main SFT dataset: mostly short/medium clean Numina
reasoning, an upper-tail Numina bucket, and an oversampled public-MCQ bucket for
boxed-letter discipline. The validation set remains Numina-only.

```bash
python -m experiments.supervised_fine_tuning.src.prepare_mixed_sft \
  --out_dir /cephfs/qwen_math_comp/data_mixed_27k_8192_publicmcq \
  --bucket_spec short:8100:150:128:800:random,medium:10800:250:800:1350:random,upper_tail:5100:100:1350:8193:longest \
  --public_mcq_path data/public.jsonl \
  --public_mcq_train 3000 \
  --max_solution_chars 0 \
  --boxed_policy single \
  --seed 42
```

The upper-tail boundary is `1350` rather than `1400` because the strict
single-boxed Numina filter produced 5,174 usable rows at `>=1400`, just short of
the requested 5,200 train+val examples.

Built dataset stats:

- train: 27,000 rows (`short=8100`, `medium=10800`, `upper_tail=5100`, `public_mcq=3000`)
- val: 500 Numina-only rows (`short=150`, `medium=250`, `upper_tail=100`)
- train token range: 129-4,134
- val token range: 164-3,084
- public MCQ base rows: 375 usable, oversampled to 3,000 train rows

Train:

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_mixed_27k_8192_publicmcq/sft_train_27k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_mixed_27k_8192_publicmcq/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq \
  --max_seq_len 8192 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10
```

Short training smoke:

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_mixed_27k_8192_publicmcq/sft_train_27k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_mixed_27k_8192_publicmcq/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/debug_mixed_27k_8192_20steps \
  --max_seq_len 8192 \
  --max_steps 20 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 20 \
  --save_steps 20 \
  --logging_steps 1
```

A 5-step smoke completed successfully at:

```text
/cephfs/qwen_math_comp/outputs/debug_mixed_27k_8192_5steps/final_adapter
```

Smoke metrics: `eval_loss=0.7167`, `train_loss=0.7018`, no dataset rows skipped
for length or malformed messages.

Final adapter will be written to:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter
```

If the run dies, restart the same command with `--resume_from_checkpoint` added.
With `--save_steps 100`, the first resumable checkpoint appears at step 100:

```bash
python train_qlora_sft.py \
  --train_path /cephfs/qwen_math_comp/data_mixed_27k_8192_publicmcq/sft_train_27k.jsonl \
  --eval_path /cephfs/qwen_math_comp/data_mixed_27k_8192_publicmcq/sft_val_500.jsonl \
  --output_dir /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq \
  --max_seq_len 8192 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 10 \
  --resume_from_checkpoint
```

To resume a specific checkpoint instead of the latest one:

```bash
python train_qlora_sft.py ... \
  --resume_from_checkpoint /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/checkpoint-100
```

## Eval After Training

```bash
export WANDB_NAME=sft_numina10k_harder_sc_terse_4096_vllm

python -m experiments.prompt_engineering.src.eval \
  run=sc_terse_only \
  eval.max_tokens=4096 \
  runner.engine=vllm \
  runner.quant=bf16 \
  runner.adapter_path=/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder/final_adapter \
  results_dir=/cephfs/qwen_math_comp/eval_results \
  run_name=sft_numina10k_harder_sc_terse_4096_vllm
```

For the 20k run, change `runner.adapter_path`, `WANDB_NAME`, and `run_name`
from `10k` to `20k`.

For the 32k-token SFT run, also raise the eval generation cap:

```bash
export WANDB_NAME=sft_numina20k_32k_sc_terse_32768_vllm

python -m experiments.prompt_engineering.src.eval \
  run=sc_terse_only \
  eval.max_tokens=32768 \
  runner.engine=vllm \
  runner.quant=bf16 \
  runner.adapter_path=/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_20k_32k/final_adapter \
  results_dir=/cephfs/qwen_math_comp/eval_results \
  run_name=sft_numina20k_32k_sc_terse_32768_vllm
```

For the mixed 27k adapter:

```bash
export WANDB_NAME=sft_mixed_27k_8192_publicmcq_sc_terse_32768_vllm

python -m experiments.prompt_engineering.src.eval \
  run=sc_terse_only \
  eval.max_tokens=32768 \
  runner.engine=vllm \
  runner.quant=bf16 \
  runner.adapter_path=/cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  results_dir=/cephfs/qwen_math_comp/eval_results \
  run_name=sft_mixed_27k_8192_publicmcq_sc_terse_32768_vllm
```

For `cot_structured` instead of `sc_terse`:

```bash
export WANDB_NAME=sft_mixed_27k_8192_publicmcq_cot_structured_32768_vllm

python -m experiments.prompt_engineering.src.eval \
  run=cot_structured_only \
  eval.max_tokens=32768 \
  runner.engine=vllm \
  runner.quant=bf16 \
  runner.adapter_path=/cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  results_dir=/cephfs/qwen_math_comp/eval_results \
  run_name=sft_mixed_27k_8192_publicmcq_cot_structured_32768_vllm
```

On DataHub or any machine where vLLM fails, keep using the HF/BnB fallback:

```bash
export ADAPTER_PATH=/cephfs/qwen_math_comp/outputs/qwen3_4b_numina_10k_harder/final_adapter
export WANDB_NAME=sft_numina10k_harder_sc_terse_4096_hf

python -m experiments.prompt_engineering.src.eval \
  run=sc_terse_only \
  eval.max_tokens=4096 \
  runner.engine=hf \
  runner.quant=bnb \
  results_dir=/cephfs/qwen_math_comp/eval_results \
  run_name=sft_numina10k_harder_sc_terse_4096_hf
```

## Google Drive Upload with rclone

Use this after the full SFT run has produced:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter
```

The Colab notebook expects the adapter at:

```text
/content/drive/MyDrive/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter
```

Set up rclone once:

```bash
# Option A: if rclone is already installed, skip this.
curl https://rclone.org/install.sh | sudo bash

# Option B: no sudo, if conda/mamba is available.
conda install -c conda-forge rclone -y

rclone config
```

In the interactive config:

```text
n
name> gdrive
Storage> drive
client_id>        # press Enter
client_secret>    # press Enter
scope> 1
root_folder_id>   # press Enter
service_account_file> # press Enter
Edit advanced config? n
Use auto config? y
Configure this as a Shared Drive? n
```

If the machine is headless and cannot open a browser, answer `n` for auto
config, then run the printed `rclone authorize "drive" ...` command on a local
machine with a browser and paste the token back into the server prompt.

Verify the remote:

```bash
rclone lsd gdrive:
rclone mkdir gdrive:qwen_math_comp
```

Upload the final mixed adapter to the exact Drive path used by the notebook:

```bash
rclone copy \
  /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  gdrive:qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  --progress \
  --transfers 8 \
  --checkers 16

rclone check \
  /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  gdrive:qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  --one-way
```

Upload eval results after running either shell eval or the notebook:

```bash
rclone copy \
  /cephfs/qwen_math_comp/eval_results/sft_mixed_27k_8192_publicmcq_cot_structured_32768_vllm \
  gdrive:qwen_math_comp/eval_results/sft_mixed_27k_8192_publicmcq_cot_structured_32768_vllm \
  --progress \
  --transfers 8 \
  --checkers 16
```

Useful inspection commands:

```bash
rclone ls gdrive:qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter
rclone ls gdrive:qwen_math_comp/eval_results
rclone about gdrive:
```
