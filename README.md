# KoRAE

<p align="center"><img src="/assets/KoRAE.png", width='256', height='256'></p>

We introduce **KoRAE** which finetuned with filtered high-quality Korean dataset.

The **KoRAE** is output of combination of high-quality data which filtered by special data filtering method and Korean Llama-2 that Korean vocabularis were added. 
We utilized special data filtering methods which introduced in [AlpaGasus](https://arxiv.org/abs/2307.08701) to filter high-quality data from mixture of several Korean datasets(OpenOrca-KO, KOpen-Platypus, KoCoT_2000, databricks-dolly-15k-ko). 
Thanks to [@kyujinpy](https://huggingface.co/kyujinpy) and [@nlp-ai](https://huggingface.co/nlpai-lab) for providing Korean datasets.
We finetuned [Korean Llama-2](https://huggingface.co/beomi/llama-2-koen-13b) that introduced by [@beomi](https://huggingface.co/beomi).

The KoRAE will be uploaded in [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)!
Stay tuned for the update of KoRAE!

The model and dataset are available via HuggingFace: [Cartinoe5930](https://huggingface.co/Cartinoe5930)

## Data Process

```
python 
```

**Composition of KoRAE dataset**
- [OpenOrca-KO](https://huggingface.co/datasets/kyujinpy/OpenOrca-KO)
- [KOpen-Platypus](https://huggingface.co/datasets/kyujinpy/KOpen-platypus)
- [KoCoT_2000](https://huggingface.co/datasets/kyujinpy/KoCoT_2000)
- [databricks-dolly-15k-ko](https://huggingface.co/datasets/nlpai-lab/databricks-dolly-15k-ko)

## Setup

Since this repository is multi-GPU friendly because of model or data parallelism, we suggest to use multi-GPU.
The initial setup of KoRAE can be done as follows. 

```
cd KoRAE
pip install -r requirements.txt
```

## Translation

```
cd KoRAE
python data_pipeline/deepl_translate.py \
    --API_KEY your_deepl_API_KEY \
    --data_type hf \
    --data_path StudentLLM/Open-Wyvern-74k
```

## Finetuning

We finetuned KoRAE with 2 * A100 80G GPUs.
In addition, we used falsh-attention-2 for the efficient and fast training.
You can utilize DeepSpeed optionally by running the code with `accelerate launch` and `accelerate_config` files for faster fine-tuning.
The hyperparameters used for finetuning of KoRAE are as follows:

**Hyperparameters**
|Hyperparameters|Value|
|---|---|
|Dataset|Cartinoe5930/KoRAE_filtered_12k|
|Batch size|128|
|Micro batch size|8|
|Epochs|3|
|Learning rate|3e-4|
|lr_scheduler|cosine|
|Max length|4096|
|Warmup ratio|0.03|
|bf16|True|

To run `finetuning/finetune.py`, please refer to following codes.

**Llama-2**

- torchrun version
```
torchrun --nproc_per_node=GPU_NUMES finetuning/finetune.py \
    --base_model beomi/llama-2-koen-13b \
    --data_path Cartinoe5930/KoRAE_filtered_12k \
    --output_dir finetuning/result/llama2/ \
    --wandb_project KoRAE_llama2 \
    --wandb_run_name KoRAE_llama2 \
    --hub_path HUB_PATH_TO_UPLOAD_MODEL \
    --auth_token YOUR_HF_ACCESS_TOKEN \
```

- accelerate version
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMS finetuning/finetune.py \
    --base_model beomi/llama-2-koen-13b \
    --data_path Cartinoe5930/KoRAE_filtered_12k \
    --output_dir finetuning/result/llama2/ \
    --wandb_project KoRAE_llama2 \
    --wandb_run_name KoRAE_llama2 \
    --hub_path HUB_PATH_TO_UPLOAD_MODEL \
    --auth_token YOUR_HF_ACCESS_TOKEN \
```

**Polyglot-ko**

- torchrun version
```
torchrun --nproc_per_node=GPU_NUMES finetuning/finetune.py \
    --base_model EleutherAI/polyglot-ko-12.8b \
    --data_path Cartinoe5930/KoRAE_filtered_12k \
    --output_dir finetuning/result/polyglot/ \
    --wandb_project KoRAE_polyglot \
    --wandb_run_name KoRAE_polyglot \
    --hub_path HUB_PATH_TO_UPLOAD_MODEL \
    --auth_token YOUR_HF_ACCESS_TOKEN \
```

- accelerate version
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMS finetuning/finetune.py \
    --base_model EleutherAI/polyglot-ko-12.8b \
    --data_path Cartinoe5930/KoRAE_filtered_12k \
    --output_dir finetuning/result/polyglot/ \
    --wandb_project KoRAE_polyglot \
    --wandb_run_name KoRAE_polyglot \
    --hub_path HUB_PATH_TO_UPLOAD_MODEL \
    --auth_token YOUR_HF_ACCESS_TOKEN \
```
