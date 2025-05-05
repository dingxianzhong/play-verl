# VERL Experiments with Qwen Models

This repository contains reproducible training scripts, benchmarking results, and insights from running **VERL** (Volcengine Reinforcement Learning) with **Qwen2.5-0.5B** and **Qwen2-7B-Instruct** on the **GSM8K** dataset using 8×H100-80GB GPUs.

The objective is to explore PPO and GRPO training in VERL, evaluate trade-offs across different configurations.

---

## 🔍 Overview

VERL ([https://github.com/volcengine/verl](https://github.com/volcengine/verl)) is a modular, high-performance RL training framework. This repo shows:

* How to fine-tune LLMs with PPO and GRPO in VERL
* How rollout size, token length, offloading, and TP affect memory, throughput, and reward
* How to optimize VERL for different resource constraints (e.g., min-memory vs. high-throughput)
* Scripts to run the experiments.

---

## 📁 Directory Structure

```
play-verl/
│
├── examples/
│   └── tuning/
│       ├── run_demo_qwen2.5b_grpo_8gpu.sh        # Qwen2.5-0.5B, GRPO baseline
│       ├── run_demo_qwen2.5b_ppo_8gpu.sh         # Qwen2.5-0.5B, PPO baseline
│       ├── qwen2_7b_gsm8k_8gpu_min.sh            # Qwen2-7B, PPO, minimal config
│       └── qwen2_7b_gsm8k_8gpu_recommended.sh    # Qwen2-7B, PPO, preferred config
├── logs/                                         # Training logs
│   ├── verl_grpo_qwen2.5b_8gpu.log
│   ├── verl_ppo_qwen2.5b_8gpu.log
│   ├── verl_qwen2_7b_min.log
│   └── verl_qwen2_7b_recommended.log
│
└── README.md
```

---

## 📊 Experiment Results & Insights

All experiments used 8×H100 GPUs on the GSM8K dataset. Key goals were to:

* Benchmark PPO vs. GRPO (small model)
* Compare minimal vs. high-throughput PPO configs (large model)
* Evaluate how rollout size, TP, and offloading impact accuracy and performance

> 📈 WandB logs:
>
> * [GRPO - Qwen2.5-0.5B](https://wandb.ai/xianzhong/verl_grpo_qwen2.5b_gsm8k)
> * [PPO - Qwen2.5-0.5B](https://wandb.ai/xianzhong/verl_ppo_qwen2.5_8gpu)
> * [PPO - Qwen2-7B Min](https://wandb.ai/xianzhong/verl_qwen2_7b_gsm8k?run=ppo_qwen2_7b_min)
> * [PPO - Qwen2-7B Recommended](https://wandb.ai/xianzhong/verl_qwen2_7b_gsm8k?run=ppo_qwen2_7b_recommended)

| ID | Model        | Algorithm | TP | Rollout.n | Dynamic BSZ | Token Len | Test Score | Script Name                          | Throughput (tok/s) | Mem (GB) | Notes                            |
| -- | ------------ | --------- | -- | --------- | ----------- | --------- | ---------- | ------------------------------------ | ------------------ | -------- | -------------------------------- |
| #1 | Qwen2.5-0.5B | GRPO      | 1  | 2         | ✅           | 8000      | 0.5155     | `run_demo_qwen2.5b_grpo_8gpu.sh`     | 461.9              | 65.7     | GRPO improves reward stability   |
| #2 | Qwen2.5-0.5B | PPO       | 1  | 1         | ❌           | 16384     | 0.4989     | `run_demo_qwen2.5b_ppo_8gpu.sh`      | 1322.3             | 57.9     | PPO runs faster but less stable  |
| #3 | Qwen2-7B     | PPO       | 1  | 2         | ✅           | 8192      | 0.8211     | `qwen2_7b_gsm8k_8gpu_min.sh`         | 201.2              | 69.2     | Efficient config with offloading |
| #4 | Qwen2-7B     | PPO       | 2  | 5         | ✅           | 12000     | 0.8810     | `qwen2_7b_gsm8k_8gpu_recommended.sh` | 892.8              | 83.6     | Best throughput and reward       |

### 💡 Key Observations

* **GRPO vs PPO on Qwen2.5-0.5B**:

  * GRPO is more reward-stable under rollout.n=2.
  * PPO achieves higher throughput but slightly lower reward.

* **Minimal vs Recommended PPO on Qwen2-7B**:

  * `qwen2_7b_gsm8k_8gpu_recommended.sh` boosts throughput 4.4× vs. the minimal version.
  * Larger rollout.n and TP=2 improve final reward from 0.8211 → 0.8810.
  * Disabling offloading increases memory use but improves compute efficiency.

* **Dynamic BSZ** is crucial for handling long sequences and GPU load balancing.

* **Token Length** tuning strongly impacts memory — 16384 in PPO-0.5B vs. 8192/12000 in PPO-7B.

---

## ⚙️ Contributions

* Four reproducible training scripts with varied configurations
* Clear comparison of PPO and GRPO behavior on LLMs
* Practical experience tuning rollout.n, TP, offloading, and token limits
* Real-world benchmarking with metrics (reward, throughput, memory)


---

## 🧠 System Setup

* **GPUs**: 8× NVIDIA H100 (80GB)
* **Models**: Qwen2.5-0.5B and Qwen2-7B-Instruct
* **Inference**: vLLM 0.8.3
* **Training**: FSDP + PPO/GRPO
* **Dataset**: GSM8K (Parquet format)

---

