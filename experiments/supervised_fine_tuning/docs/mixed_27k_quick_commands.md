# Mixed 27k SFT Quick Commands

Use these from the repo root:

```bash
cd /workspace/151B_SP26_Competition
```

## Start Training

This trains the mixed 27k dataset with `max_seq_len=8192` and saves checkpoints
every 100 optimizer steps.

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

Final adapter path:

```text
/cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter
```

## Run Inside tmux

```bash
tmux new -s sft_mixed_27k
cd /workspace/151B_SP26_Competition
```

Then run the training command above. Detach with `Ctrl-b`, then `d`.

Reattach:

```bash
tmux attach -t sft_mixed_27k
```

## Check Checkpoints

The first resumable checkpoint appears at step 100 because `--save_steps=100`.

```bash
ls -d /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/checkpoint-* 2>/dev/null | sort -V
```

Latest checkpoint:

```bash
ls -d /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/checkpoint-* 2>/dev/null | sort -V | tail -1
```

## Resume Training

Resume from the latest checkpoint automatically:

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

Resume from a specific checkpoint:

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
  --resume_from_checkpoint /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/checkpoint-100
```

## Eval After Training

`cot_structured` with long-budget inference:

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

`sc_terse` with long-budget inference:

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

## rclone Setup

Install rclone, using one option:

```bash
curl https://rclone.org/install.sh | sudo bash
```

```bash
conda install -c conda-forge rclone -y
```

Create a Google Drive remote named `gdrive`:

```bash
rclone config
```

Interactive answers:

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

If the server is headless, answer `n` for `Use auto config?`, then run the
printed `rclone authorize "drive" ...` command on a local machine with a browser
and paste the token back into the server prompt.

Verify:

```bash
rclone lsd gdrive:
rclone mkdir gdrive:qwen_math_comp
```

## Upload Adapter to Google Drive

This is the path expected by the Colab notebook:

```text
/content/drive/MyDrive/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter
```

Upload:

```bash
rclone copy \
  /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  gdrive:qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  --progress \
  --transfers 8 \
  --checkers 16
```

Verify upload:

```bash
rclone check \
  /cephfs/qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  gdrive:qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter \
  --one-way
```

Inspect:

```bash
rclone ls gdrive:qwen_math_comp/outputs/qwen3_4b_mixed_27k_8192_publicmcq/final_adapter
rclone about gdrive:
```

## Upload Eval Results

```bash
rclone copy \
  /cephfs/qwen_math_comp/eval_results/sft_mixed_27k_8192_publicmcq_cot_structured_32768_vllm \
  gdrive:qwen_math_comp/eval_results/sft_mixed_27k_8192_publicmcq_cot_structured_32768_vllm \
  --progress \
  --transfers 8 \
  --checkers 16
```
