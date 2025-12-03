# VERL Project Overview

## Purpose
VERL (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training library for large language models (LLMs). It is the open-source version of the HybridFlow framework presented in the [EuroSys 2025 paper](https://arxiv.org/abs/2409.19256v2).

## Key Characteristics
- **Initiated by**: ByteDance Seed team
- **Maintained by**: verl community
- **License**: Apache 2.0
- **Python Version**: >=3.10
- **Primary Language**: Python

## Core Value Propositions

### Flexibility
- **Easy extension of RL algorithms**: Hybrid-controller programming model enables flexible representation and efficient execution of complex post-training dataflows
- **Seamless integration**: Decouples computation and data dependencies, enabling integration with existing LLM frameworks (FSDP, Megatron-LM, vLLM, SGLang)
- **Flexible device mapping**: Various placement of models onto different GPU sets for efficient resource utilization

### Performance
- **State-of-the-art throughput**: SOTA LLM training and inference engine integrations
- **Efficient actor model resharding**: 3D-HybridEngine eliminates memory redundancy and reduces communication overhead during training/generation transitions

## Main Features

### Training Backends
- FSDP, FSDP2, Megatron-LM

### Inference Backends  
- vLLM, SGLang, HF Transformers

### Supported Models
- Compatible with Hugging Face Transformers and Modelscope Hub
- Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc.
- Vision-language models (Qwen2.5-vl, Kimi-VL)

### RL Algorithms
PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, KL_Cov, Clip_Cov, CPO

### Additional Features
- Supervised fine-tuning
- Model-based and function-based rewards (for math, coding)
- Multi-modal RL
- Multi-turn with tool calling
- Flash attention 2, sequence packing, sequence parallelism
- LoRA support
- Scales to 671B models with expert parallelism
- Experiment tracking (wandb, swanlab, mlflow, tensorboard)

## Notable Achievements
- Powers Doubao-1.5-pro (70.0 pass@1 on AIME)
- DAPO achieves 50 points on AIME 2024 (Qwen2.5-32B base)
- Seed-Thinking-v1.5: 86.7 on AIME 2024
- VAPO achieves 60.4 on AIME 2024
