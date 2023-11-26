# KoRAE

<p align="center"><img src="/assets/KoRAE.png", width='256', height='256'></p>

We introduce **KoRAE** which finetuned with filtered high-quality Korean dataset.

The **KoRAE** is output of combination of high-quality data which filtered by special data filtering method and Korean Llama-2 that Korean vocabularis were added. 
We utilized special data filtering methods which introduced in [AlpaGasus](https://arxiv.org/abs/2307.08701) to filter high-quality data from mixture of several Korean datasets(OpenOrca-KO, KOpen-Platypus, KoCoT_2000, databricks-dolly-15k-ko). 
We finetuned [Korean Llama-2](https://huggingface.co/beomi/llama-2-koen-13b) that introduced by [@beomi](https://huggingface.co/beomi) on the filtered dataset.
The Flash-Attention2 and LoRA were utilized for efficient finetuning.

The KoRAE will be uploaded in [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)!
Stay tuned for the update of KoRAE!

The model and dataset are available via HuggingFace: [Cartinoe5930](https://huggingface.co/Cartinoe5930)

## Setup

This repository mainly uses Transformers and TRL provided by HuggingFace🤗. Please keep in mind!
In addition, Flash Attention 2 and LoRA are used for the Parameter Efficient Fine Tuning(PEFT).

```
cd KoRAE
pip install -r requirements.txt
```

## Dataset

We used filtered high-quality Korean dataset for finetuning as mentioned above.
First of all, we gathered Korean data and make the mixture of them.
Then we filtered high-quality data from the combination of data through filtering method that introduced from [AlpaGasus](https://arxiv.org/abs/2307.08701).
The overview of data processing procedure is as follws:

1. Collect various Korean dataset from HuggingFace Hub.
2. Rate the data quality using `gpt-3.5-turbo`.
3. Process the rated data and filter the high-scored data.

Let's go deeper into data processing!

### 1. Korean dataset mixture

We investigated several sources to collect high-quality Korean data, and among them, we collected data from the various sources.
As a result, we were able to create a new dataset containing 64K pieces of data.
The specific configuration of the dataset is as follows:

|Dataset|# Nums|
|---|---|
|**[OpenOrca-ko](https://huggingface.co/datasets/kyujinpy/OpenOrca-KO)**|21.6k|
|**[KOpen-Platypus](https://huggingface.co/datasets/kyujinpy/KOpen-platypus)**|24.9k|
|**[KoCoT_2000](https://huggingface.co/datasets/kyujinpy/KoCoT_2000)**|2.1k|
|**[databricks-dolly-15k-ko](https://huggingface.co/datasets/nlpai-lab/databricks-dolly-15k-ko)**|15k|
|**Total**|63.7k|

Thanks to [@kyujinpy](https://huggingface.co/kyujinpy) and [@nlp-ai](https://huggingface.co/nlpai-lab) for providing Korean datasets.

### 2. Rating

We utilized ChatGPT(gpt-3.5-turbo) as rater to rate the quality of dataset.
We considered whether to use the prompt for the evaluation in Korean or English, but we thought it would be undesirable to give evaluations in different languages, so we conducted the evaluation using the Korean prompt.
However, since the rating code `rating/rating.py` also support English rating prompt format, you can choose the rating mode according to your preference.

**Korean version**
```
python rating/rating.py \
    --i 0 \
    --rating_type ko \
    --api_key YOUR_OPENAI_KEY
```

**English version**
```
python rating/rating.py \
    --i 0 \
    --rating_type en \
    --api_key YOUR_OPENAI_KEY
```

The rating code `rating/rating.py` and rating prompt format `teamplates/rating_template.json` were referred to [AlpaGasus](https://github.com/gpt4life/alpagasus)

### 3. Processing & Filtering

We postprocessed rated dataset after the rating.
The main postprocessing procedure are as follows:

- Wrong score extraction correction
- Incorrect format dataset exclusion

After the all postprocessing, we analysed the score distribution of rated dataset.
As shown in the following figure, it was confirmed that 8-point data was the most.
This confirms that KoRAE dataset consisted of high-quality data from the beginning.

<img src="/assets/rated_dataset_distribution.png">

However, We filtered data only with a score of 8.5 or higher and used it to finetune KoRAE for better performance.
As a result, we were able to filter the dataset 64k to 12k!

```
python rating/filtering.py \
    --score_criteria 8.5 \
    --output_dir PATH_TO_UPLOAD_DATASET \
    --hf_token YOUR_HF_ACCESS_TOKEN
```

The original and filtered dataset are uploaded on HuggingFace Hub, so you can check them! 

- [Cartinoe5930/KoRAE_original](https://huggingface.co/datasets/Cartinoe5930/KoRAE_original)
- [Cartinoe5930/KoRAE_filtered_12k](https://huggingface.co/datasets/Cartinoe5930/KoRAE_filtered_12k)

## Finetuning(SFT)

We finetuned KoRAE with  Flash-Attention2 and LoRA for efficient finetuning on 1 * A100 80G GPUs.
KoRAE was finetuned with Parameter Efficient Fine Tuning method, which called LoRA.
In addition, since the high-quality filtered dataset is smaller than the original dataset, it makes finetuning more efficient.
As a result, it took only 5 GPU hours to fully finetune the model with 3 epochs! 
The hyperparameters used for finetuning of KoRAE are as follows:

**Training Hyperparameters**
|Hyperparameters|Value|
|---|---|
|**Base model**|beomi/llama-2-koen-13b|
|**Dataset**|Cartinoe5930/KoRAE_filtered_12k|
|**Batch size**|16|
|**Micro batch size**|4|
|**Gradient accumulation steps**|4|
|**Epochs**|3|
|**Learning rate**|1e-5|
|**lr_scheduler**|cosine|
|**Max length**|4096|
|**Warmup ratio**|0.03|
|**Weight decay**|0|
|**bf16**|True|
|**Gradient checkpointing**|True|

**LoRA Hyperparameters**
|Hyperparameters|Value|
|---|---|
|**lora_r**|8|
|**lora_alpha**|16|
|**lora_dropout**|0.05|

The finetuning code of KoRAE is as follows:

```
python finetuning/finetune_lora.py \
    --model_path beomi/llama-2-koen-13b \
    --data_path Cartinoe5930/KoRAE_filtered_12k \
    --output_dir finetuning/result/ \
    --wandb_project KoRAE_sft \
    --wandb_run_name KoRAE_sft \
    --hf_hub_path HUB_PATH_TO_UPLOAD_MODEL \
    --hf_token YOUR_HF_ACCESS_TOKEN
```

## DPO

We additionally trained KoRAE with DPO for improving the model.
Since we need binarized feedback to train the model with DPO, we utilized the [ko_Ultrafeedback_binarized](https://huggingface.co/datasets/maywell/ko_Ultrafeedback_binarized) which is the Korean translated version of [Ultrafeedback_binarized]() provided by [@maywell](https://huggingface.co/maywell).
The hyperparameters used for DPO training of KoRAE are as follows and LoRA hyperparameters are the same as mentioned above:


**DPO Hyperparameters**
|Hyperparameters|Value|
|---|---|
|**Beta**|0.1|
|**Batch size**|16|
|**Micro batch size**|4|
|**Gradient accumulation steps**|4|
|**Epochs**|3|
|**Learning rate**|5e-7|
|**lr_scheduler**|linear|
|**Max prompt length**|2048|
|**Max length**|4096|
|**Warmup ratio**|0.1|
|**Weight decay**|0|
|**Gradient checkpointing**|True|

The DPO training code of KoRAE is as follows:

```
python DPO/dpo.py \
    --model_path Cartinoe5930/KoRAE-13b \
    --data_path maywell/ko_Ultrafeedback_binarized \
    --output_dir DPO/result/ \
    --wandb_project KoRAE_dpo \
    --wandb_run_name KoRAE_dpo \
    --hf_hub_path HUB_PATH_TO_UPLOAD_MODEL \
    --hf_token YOUR_HF_ACCESS_TOKEN
```

## Prompting Format

We utilized following prompt format for KoRAE.
To follow the prompting format of popular models and preserve important information introduced in instruction, we used it.
You can check the prompting format of KoRAE in `templates/KoRAE_template.json` or the following example:

```
### System:
{system_prompt}

### User:
{instruction + input}

### Assistant:
{output}
```

Since we implemented KoRAE prompt format on model's tokenizer, you can utilize it with `apply_chat_template`.
For more details, please refer to the [Model card](https://huggingface.co/Cartinoe5930/KoRAE-13b) of KoRAE!

## Weights & Bias Result

The finetuning and DPO training results of KoRAE can be checked following the Weights & Bias link.

- [SFT](https://wandb.ai/kopilot100/KoRAE_llama2/runs/d16ciamc?workspace=user-kopilot100)
- [DPO]() - still training

## Open Ko-LLM Leaderboard

Stay tuned for the update of KoRAE!

## Citation

- [KO-Platypus](https://github.com/Marker-Inc-Korea/KO-Platypus)
- [Korean-OpenOrca](https://github.com/Marker-Inc-Korea/Korean-OpenOrca)
- [ko_Ultrafeedback_binarized](https://huggingface.co/datasets/maywell/ko_Ultrafeedback_binarized)

```
@inproceedings{lee2023kullm,
  title={KULLM: Learning to Construct Korean Instruction-following Large Language Models},
  author={Lee, SeungJun and Lee, Taemin and Lee, Jeongwoo and Jang, Yoona and Lim, Heuiseok},
  booktitle={Annual Conference on Human and Language Technology},
  pages={196--202},
  year={2023},
  organization={Human and Language Technology}
}
```

```
@misc{chen2023alpagasus,
      title={AlpaGasus: Training A Better Alpaca with Fewer Data}, 
      author={Lichang Chen and Shiyang Li and Jun Yan and Hai Wang and Kalpa Gunaratna and Vikas Yadav and Zheng Tang and Vijay Srinivasan and Tianyi Zhou and Heng Huang and Hongxia Jin},
      year={2023},
      eprint={2307.08701},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@misc {l._junbum_2023,
    author       = { {L. Junbum, Taekyoon Choi} },
    title        = { llama-2-koen-13b },
    year         = 2023,
    url          = { https://huggingface.co/beomi/llama-2-koen-13b },
    doi          = { 10.57967/hf/1280 },
    publisher    = { Hugging Face }
}
```