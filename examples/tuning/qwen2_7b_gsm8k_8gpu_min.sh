#!/bin/bash
# PPO - Min Memory Config for Qwen2-7B-Instruct on GSM8K with 8×A100 (80GB)
# WandB: https://wandb.ai/your_name/verl_ppo_qwen2_7b_gsm8k

set -x

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_MODE=V1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

python3 -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=512 \
  data.val_batch_size=512 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
  +actor_rollout_ref.model.dtype=bfloat16 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path=Qwen/Qwen2-7B-Instruct \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger='[console, wandb]' \
  trainer.project_name=verl_qwen2_7b_gsm8k \
  trainer.experiment_name=ppo_qwen2_7b_min \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.total_epochs=10 "$@" | tee verl_qwen2_7b_min.log
