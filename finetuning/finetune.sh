torchrun --nproc_per_node=2 finetuning/finetune.py \
    --base_model beomi/llama-2-koen-13b \
    --data_path Cartinoe5930/KoRAE_filtered_12k \
    --output_dir finetuning/result/ \
    --hub_path HUB_PATH_TO_UPLOAD_MODEL \
    --auth_token YOUR_HF_ACCESS_TOKEN \