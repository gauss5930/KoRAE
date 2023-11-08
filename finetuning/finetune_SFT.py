import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from accelerate import Accelerator
from datasets import load_dataset, Dataset

from trl import SFTTrainer
from accelerate import Accelerator

from utils.prompter import Prompter

import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, help="Required to upload models to hub.")
    parser.add_argument("--model_path", type=str, default="beomi/llama-2-koen-13b")
    parser.add_argument("--data_path", type=str, default="Cartinoe5930/KoRAE_filtered_12k")
    parser.add_argument("--num_proc", type=int)

    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--val_set_size", type=float, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch", help="You can choose the strategy of saving model.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=False)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
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
        required=True,
        help="The hub path to upload the model"
    )

    return parser.parse_args()

def process_dataset(dataset):
    prompter = Prompter("KoRAE_template")

    list_data = dataset.to_list()
    
    for data in list_data:
        data["prompted_input"] = prompter.generate_prompt(
            data["instruction"],
            data["prompt"],
            data["input"],
            data["output"])

    result_data = Dataset.from_list(list_data)

    return result_data

def create_datasets(args):
    dataset = load_dataset(
        args.data_path,
        split="train",
        num_proc=args.num_proc if args.num_proc else None,
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

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size // args.num_proc

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": Accelerator().process_index},
        torch_dtype=torch.bfloat16,
        use_auth_token=args.hf_token,
        use_flash_attention_2=True
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_auth_token=args.hf_token,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    train_dataset, eval_dataset = create_datasets(args)

    train_dataset = process_dataset(train_dataset)
    eval_dataset = process_dataset(eval_dataset) if eval_dataset else None
    
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
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else None,
        run_name=args.wandb_run_name if use_wandb else None,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="prompted_input",
        data_collator=data_collator,
        packing=args.packing,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    model.push_to_hub(
        args.hf_hub_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token,
    )
    tokenizer.push_to_hub(
        args.hf_hup_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token,
    )