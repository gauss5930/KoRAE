import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from datasets import load_dataset
import huggingface_hub

from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM

from utils.prompter import Prompter

import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, help="Required to upload models to hub.")
    parser.add_argument("--model_path", type=str, default="beomi/llama-2-koen-13b")
    parser.add_argument("--data_path", type=str, default="Cartinoe5930/KoRAE_filtered_12k")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--val_set_size", type=float, default=0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="epoch", help="You can choose the strategy of saving model.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=False, help="If True, the short sentences would be concatenated.")

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
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

    prompter = Prompter()

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_cache=not args.gradient_checkpointing,
        use_flash_attention_2=True
    )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    base_model = get_peft_model(base_model, peft_config)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "right"
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ '### System:\n' + message['content'] + eos_token }}\n\n{% elif message['role'] == 'user' %}\n{{ '### User:\n' + message['content'] + eos_token }}\n\n{% elif message['role'] == 'assistant' %}\n{{ '### Assistant:\n'  + message['content'] + eos_token }}\n\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '### Assistant:\n' }}\n{% endif %}\n{% endfor %}"

    def tokenize(prompt, args=args, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.seq_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.seq_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["prompt"],
            data_point["input"],
            data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["prompt"],
            data_point["input"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        
        return tokenized_full_prompt

    train_dataset, eval_dataset = create_datasets(args)

    original_column = train_dataset.column_names
    
    train_dataset = train_dataset.shuffle().map(generate_and_tokenize_prompt)
    train_dataset = train_dataset.remove_columns(original_column)
    
    eval_dataset = eval_dataset.shuffle().map(generate_and_tokenize_prompt) if eval_dataset else None
    eval_dataset = eval_dataset.remove_columns(original_column) if eval_dataset else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size if eval_dataset else None,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        evaluation_strategy="epoch" if eval_dataset else "no",
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else None,
        run_name=args.wandb_run_name if use_wandb else None,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

    trainer = Trainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)

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