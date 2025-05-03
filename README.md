
# VERL Experiments with Qwen Models

This repository contains a collection of experiments, reproducible scripts, performance logs, and insights from running **VERL** (Volcengine Reinforcement Learning) with **Qwen2.5-0.5B** and **Qwen2-7B-Instruct** on the **GSM8K** dataset using 8×A100-80GB GPUs.

The goal is to explore PPO and GRPO training in VERL, evaluate throughput and memory usage under different rollout and parallelization configurations, and prepare meaningful contributions for the VERL community.

---

## 🔍 Project Overview

VERL (https://github.com/volcengine/verl) is a flexible and scalable RL training framework with modular rollout and training backends. This repo demonstrates:

- How to train large language models (LLMs) using PPO and GRPO in VERL.
- Performance tuning across configurations (rollout.n, TP, KL types).
- Efficient multi-GPU training using FSDP and vLLM.
- Contribution-ready training scripts and engineering extensions.

---

## 📁 Directory Structure

```
play-verl/
│
├── examples/
│   └── tuning/
│       ├── qwen2_7b_gsm8k_8gpu_recommended.sh  # high-throughput config
│       └── qwen2_7b_gsm8k_8gpu_min.sh          # memory-efficient config
│
├── wandb_logs/
│   └── ...                                     # screenshots or exports from WandB
│
├── plots/
│   └── throughput_vs_rollout.png              # comparison plots (tokens/s, memory)
│
└── README.md                                   # this file
```

---

## 🚀 Reproducible Training Scripts

### ✅ Recommended: High Throughput (rollout.n=5, TP=2)
- **Model**: Qwen2-7B-Instruct
- **Algorithm**: PPO
- **Parallelism**: TP=2
- **Rollout Batch Size**: 5
- **Offloading**: Disabled
- **Token Length**: 12000
- **Dynamic BSZ**: Enabled

📄 Script: `examples/tuning/qwen2_7b_gsm8k_8gpu_recommended.sh`

### ✅ Minimum: Memory-Efficient (rollout.n=1, TP=1)
- **Model**: Qwen2-7B-Instruct
- **Algorithm**: PPO
- **Parallelism**: TP=1
- **Rollout Batch Size**: 1
- **Offloading**: Enabled
- **Token Length**: 8192
- **Dynamic BSZ**: Enabled

📄 Script: `examples/tuning/qwen2_7b_gsm8k_8gpu_min.sh`

---

## 📊 Experimental Results

We tracked performance metrics using WandB and present comparisons below:

| Script | Model           | Algorithm | TP | Rollout.n | Dynamic BSZ | Token Len | Throughput (tok/s) | Memory Use | Notes                        |
|--------|------------------|-----------|----|-----------|-------------|-----------|---------------------|-------------|-----------------------------|
| #1     | Qwen2.5-0.5B     | GRPO      | 1  | 2         | ✅           | 8000      |  X tok/s            | X GB        | GRPO early test             |
| #2     | Qwen2.5-0.5B     | PPO       | 1  | 1         | ❌           | —         |  X tok/s            | X GB        | PPO baseline                |
| #3     | Qwen2-7B         | PPO       | 1  | 2         | ✅           | 8192      |  X tok/s            | X GB        | Balanced config             |
| #4     | Qwen2-7B         | PPO       | 2  | 5         | ✅           | 12000     |  X tok/s            | X GB        | Best throughput, preferred  |

📌 See `plots/throughput_vs_rollout.png` for visual comparison.

---

## 🛠 Contributions

To demonstrate engineering initiative, this repo includes:

- ✅ Two training scripts (`min`, `recommended`) for VERL’s example gallery
- 🧠 Planned extension: add `log_gpu_memory()` utility to VERL
- 📦 Ready-to-submit PR format (to `examples/tuning/`)
- 🔁 Potential reward function extension module

---

## 🧠 System Setup

- **GPUs**: 8× NVIDIA A100 (80GB)
- **Model Sizes**: Qwen2.5-0.5B and Qwen2-7B-Instruct
- **Inference Backend**: vLLM 0.8.3
- **Training Backend**: FSDP (no offload for recommended)
- **Dataset**: GSM8K (converted to Parquet format)

---

## 📮 Contact

Prepared by **Xianzhong Ding** for VERL maintainers.

If you'd like a deeper dive into results, experiment details, or to review a PR draft for VERL, please reach out!

---

## 🧩 TODO

- [ ] Add log_gpu_memory utility module to VERL
- [ ] Prepare PR for `examples/tuning/`
- [ ] Extend PPO reward shaping function
- [ ] Evaluate additional rollout backends (SGLang)
