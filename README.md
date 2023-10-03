# KO-Wyvern

## Plan

목표는 [open-ko-llm-leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)에서 1등을 차지하는 것으로 하자.

- base model: [Llama2-ko-7b](beomi/llama-2-ko-7b) / [polyglot-ko-12.8b](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)
- Training: 일단 지금은 빠르게 training해서 leaderboard에 업로드하는 것이 좋을 것 같으니, LoRA fine-tuning을 활용해보자. 최종적으로는 merge도 진행해야 함.
- Dataset: [Open-Wyvern-74k](https://huggingface.co/datasets/StudentLLM/Open-Wyvern-74k) 사용 예정. 다른 데이터셋도 함께 사용해보는 것도 고민해보았으나 유니크함을 보여주기 위해서는 이 데이터셋만 사용하는 것이 좋을 것 같음.

## Translation

```
cd KO-Wyvern
python data_pipeline/deepl_translate.py \
    --API_KEY your_deepl_API_KEY \
    --data_type hf \
    --data_path StudentLLM/Open-Wyvern-74k
```

## Fine-tuning

**Llama2**
```
python finetuning/finetune.py \
    --model llama \
    --data_type hf \
    --dataset ko-wyvern \
    --output_dir your_output_directory \
```

**Polyglot**
```
python finetuning/finetune.py \
    --model polyglot \
    --data_type hf \
    --dataset ko-wyvern \
    --output_dir your_output_directory \
```