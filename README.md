# Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference

<p align="center">
    <img src="img/intro.png" width="500" height="auto">
</p>

[**Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference**](https://arxiv.org/abs/2412.12785) by
[Siyuan Wang](https://siyuanwangw.github.io/), 
[Dianyi Wang](https://scholar.google.com/citations?hl=zh-CN&user=iP2HPFEAAAAJ), 
Chengxing Zhou,
[Zejun Li](https://scholar.google.com/citations?user=FYppLbUAAAAJ&hl=zh-CN),
[Zhihao Fan](https://libertfan.github.io/),
[Xuanjing Huang](https://xuanjing-huang.github.io/), and
[Zhongyu Wei](https://scholar.google.com/citations?user=AjLDxxgAAAAJ&hl=zh-CN).

> **Abstract.** Large Vision-Language Models (LVLMs) typically learn visual capacity through visual instruction tuning, involving updates to both a
projector and their LLM backbones. Inspired
by the concept of a visual region in the human
brain, we investigate the existence of an analogous visual region within LLMs that functions
as a cognitive core, and explore the potential
of efficient training of LVLMs via selective
layers tuning. Using Bunny-Llama-3-8B-V
for detailed analysis and other three LVLMs
for validation across diverse visual and textual
tasks, we find that selectively updating 25%
of LLMs layers, when sparsely and uniformly
distributed, can preserve nearly 99% of visual
performance and maintain or improve textual
task results, while effectively reducing training
time. Based on this targeted training approach,
we further propose a novel visual region-based
pruning paradigm, removing non-critical layers
outside the visual region, which can achieve
minimal performance loss. This study offers an
effective and efficient strategy for LVLM training and inference by activating a layer-wise
visual region within LLMs, which proves consistently effective across different models.


## Release

- [2025/05/15] 🔥🔥🔥 **Our paper has been accepted by ACL 2025!** 🔥🔥🔥
- [2024/12/17] 🔥 we release our paper, checkout the [paper](https://arxiv.org/pdf/2410.09575) for details.

## Contents
- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)

### Install
* CUDA and cuDNN

  We use CUDA 11.8 and cuDNN 8.7.0. We actually use the CUDA docker by NVIDIA: `docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`. CUDA 12 is fine, too.

* Create a conda virtual environment and activate it:

  ```shell
  conda create -n bunny python=3.10
  conda activate bunny
  ```


  ```shell
  pip install --upgrade pip  # enable PEP 660 support
  ```

* Install apex

  ```shell
  # https://github.com/NVIDIA/apex#from-source
  pip install ninja
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```

* Install flash-attention

  ```shell
  # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
  pip install packaging
  pip install flash-attn --no-build-isolation
  ```

* Install bunny and other requirements

  ```shell
  cd Bunny
  pip install -e .
  ```
### Training

Our model is trained on 8 A100 GPUs. Under other circumstances, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `global_batch_size ` = `per_device_train_batch_size` $`\times`$ `gradient_accumulation_steps` $`\times`$ `num_gpus`.

* Experiments model components

the "siglip-so400m-patch14-384" is for Bunny-Llama-3-8B-V and Bunny-Phi3-mini-4B-V, the "clip-336" is for LLaVA-1.5-7B/13B


| Vision Encoders            | Download Link                                                |
| -------------------------- | ------------------------------------------------------------ |
| siglip-so400m-patch14-384  | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |
| clip-336  | [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) |



| MODEL_TYPE | LLM             | Download Link                                                |
| ---------- | --------------- | ------------------------------------------------------------ |
| phi-3 | Phi-3-mini-4k-instruct | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| llama3-8b | Meta-Llama-3-8B-Instruct | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| llava | vicuna-7b-v1.5 | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| llava | vicuna-13b-v1.5 | [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |


The training script for activating visual region stage with DeepSpeed ZeRO-3 can be found in ```scripts/train/finetune_all_moe.sh```. Global Batch Size is 128

