import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
import huggingface_hub

from trl import DPOTrainer

import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, help="Required to upload models to hub.")
    parser.add_argument("--model_path", type=str, default="Cartinoe5930/KoRAE-13b")
    parser.add_argument("--data_path", type=str, default="maywell/ko_Ultrafeedback_binarized")

    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--val_set_size", type=float, default=0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="epoch", help="You can choose the strategy of saving model.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--group_by_length", type=bool, default=False)

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)
    
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--hf_hub_path",
        type=str,
        help="The hub path to upload the model"
    )

    return parser.parse_args()

def create_datasets(args):
    dataset = load_dataset(
        args.data_path,
        split="train"
    )

    if args.val_set_size > 0:
        train_val = dataset.train_test_split(test_size=args.val_set_size, seed=42)

        train_data = train_val["train"]
        val_data = train_val["test"]
    else:
        train_data = dataset
        val_data = None

    return train_data, val_data


if __name__ == "__main__":
    args = args_parse()

    huggingface_hub.login(args.hf_token)

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_cache=not args.gradient_checkpointing,
        use_flash_attention_2=True
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_cache=not args.gradient_checkpointing,
        use_flash_attention_2=True
    )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        ref_model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def dpo_process_data(example, tokenizer=tokenizer):
        prompt_message = tokenizer.apply_chat_template([
            {"role": "system", "content": "당신은 유용한 인공지능 비서입니다. 사용자가 몇 가지 지시가 포함된 작업을 제공합니다. 요청을 적절히 완료하는 응답을 작성하세요."},
            {"role": "user", "content": example["prompt"]}
        ], tokenize=False, add_generation_prompt=True)

        chosen_message = tokenizer.apply_chat_template([
            {"role": "assistant", "content": example["chosen"]}
        ], tokenize=False)

        rejected_message = tokenizer.apply_chat_template([
            {"role": "assistant", "content": example["rejected"]}
        ], tokenize=False)
        
        return {
            "prompt": prompt_message,
            "chosen": chosen_message,
            "rejected": rejected_message
        }

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    train_dataset, eval_dataset = create_datasets(args)
    original_columns = train_dataset.column_names

    train_dataset = train_dataset.map(
        dpo_process_data,
        remove_columns=original_columns
    )
    eval_dataset = eval_dataset.map(
        dpo_process_data,
        remove_columns=original_columns
    ) if eval_dataset else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size if eval_dataset else None,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        evaluation_strategy="epoch" if eval_dataset else "no",
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        save_total_limit=2 if args.save_strategy != "no" else None,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else None,
        run_name=args.wandb_run_name if use_wandb else None,
    )

    dpo_trainer = DPOTrainer(
        model=base_model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )
    
    dpo_trainer.train()

    dpo_trainer.model.save_pretrained(args.output_dir)

    del base_model
    torch.cuda.empty_cache()
    
    model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    if args.hf_hub_path:
        model.push_to_hub(args.hf_hub_path)
        tokenizer.push_to_hub(args.hf_hub_path)
    else:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)