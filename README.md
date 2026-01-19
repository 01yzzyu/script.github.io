# Script: Graph-Structured and Query-Conditioned Semantic Token Pruning for Multimodal Large Language Models
[📄 [Paper](https://openreview.net/forum?id=F6xKzbgcHq)] [🎞️ [Project Page](https://01yzzyu.github.io/script.github.io/)]

## 👁️ Overview

The rapid growth of visual tokens in multimodal large language models (MLLMs) leads to excessive memory consumption and inference latency, especially when handling high-resolution images and videos. Token pruning is a technique used to mitigate this issue by removing redundancy, but existing methods often ignore relevance to the user query or suffer from the limitations of attention mechanisms, reducing their adaptability and effectiveness. To address these challenges, we propose Script, a plug-and-play pruning method that requires no retraining and generalizes across diverse MLLMs. Script comprises two modules: a graph-structured pruning module that removes visually redundant tokens, and a query-conditioned semantic pruning module that preserves query-relevant visual information. Together, they enhance performance on multimodal tasks. Experiments on fourteen benchmarks across image and video understanding tasks show that Script consistently achieves higher model efficiency and predictive accuracy compared to existing pruning methods. On LLaVA-NeXT-7B, it achieves up to 6.8x prefill speedup and 10x FLOP reduction, while retaining 96.88\% of the original performance.

## ⚙️ Setup

### 🏝️ Environment

1. Clone this repository.
```bash
https://github.com/01yzzyu/script.github.io.git
cd Script
```

2. Install necessary packages.
```bash
conda create -n script python=3.10 -y
conda activate script
pip install -e .
```

3. (Optional) Install FlashAttention for further inference acceleration.
```bash
pip install flash-attn --no-build-isolation
```

### 📦️ Model

Download corresponding [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) checkpoints from [Hugging Face](https://huggingface.co/liuhaotian) 🤗:

| Version | LLM | Checkpoint |
|----------|:----------:|:-----------:|
| LLaVA-1.5 | Vicuna-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| LLaVA-1.5 | Vicuna-13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| LLaVA-1.6 (LLaVA-NeXT) | Vicuna-7B | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
| LLaVA-1.6 (LLaVA-NeXT) | Vicuna-13B | [liuhaotian/llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) |

### 📊 Data

Download each dataset according to [EVAL.md](EVAL.md).

## 📋️ Evaluation

The main implementation of script is highlighted with `script` annotations, mainly in [`llava_llama.py`](llava/model/language_model/llava_llama.py#L51), [`llava_arch.py`](llava/model/llava_arch.py#L140) and [`clip_encoder.py`](llava/model/multimodal_encoder/clip_encoder.py#L42).

We provide the evaluation scripts for each benchmark; you only need to set the remaining visual token number as the bash argument. For example, if you want to evaluate script with 128 visual tokens retained on the GQA benchmark, you can run the following command with the argument `128`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/gqa.sh 128
```

And if you want to evaluate the script with 64 visual tokens retained on the MME benchmark, you can run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh 64
```

For evaluation with the 13B LLM, you just need to replace the `CKPT` argument from `llava-v1.5-7b` to `llava-v1.5-13b` in each script. And for evaluation with LLaVA-NeXT, you can use the scripts in `./scripts/v1_6/eval`. For example, if you want to evaluate script with 32 * 5 = 320 visual tokens retained on the TextVQA benchmark, you can run the following command:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/eval/textvqa.sh 32
```

The detailed guidance for evaluation commands and online submission of each benchmark can be found in [EVAL.md](EVAL.md).

## 🎟️ License

This project is released under the [Apache 2.0 license](LICENSE).

## 🎉 Acknowledgement

We appreciate the open-source efforts of [LLaVA](https://github.com/haotian-liu/LLaVA), [Fast-MAP-DPP](https://github.com/laming-chen/fast-map-dpp) and [TRIM](https://github.com/FreedomIntelligence/TRIM).
