from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, set_peft_model_state_dict
import torch
import argparse
import sys

from utils.prompter import Prompter

model_path = {"llama": "beomi/llama-2-ko-7b", "polyglot": "EleutherAI/polyglot-ko-12.8b"}
data_path = {"ko-wyvern": ""}

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="You can choose 'llama' or 'polyglot'. If you want to use custom model, please enter the path of your custom model."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="hf",
        help="You can also utilize the JSON type dataset! If you want to use JSON type dataset, please enter 'json'."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ko-wyvern",
        help="The dataset which you want to fine-tune the model on. If tou want to use the custom dataset, please enter the path of custom dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory you want to save the model and tokenizer."
    )
    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--val_set_size", type=int, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    # lora hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=list, default=["gate_proj", "down_proj", "up_proj"])
    # llm hyperparameters
    parser.add_argument("--add_eos_token", type=bool, default=False)
    parser.add_argument("--group_by_length", type=bool, default=False)

    return parser.parse_args

if __name__ == "__main__":
    args = args_parse()

    if args.model == "llama" or args.model == "polyglot":
        model = AutoModelForCausalLM.from_pretrained(
            model_path[args.model],
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path[args.model]
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model
        )

    model = prepare_model_for_kbit_training(model)

    if args.dataset == "ko-wyvern":
        dataset = load_dataset(data_path[args.dataset], split="train")
    elif args.data_type == "hf":
        dataset = load_dataset(args.dataset, split="train")
    elif args.data_type == "json":
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        raise ValueError(f"Your dataset is not founeded... Is your dataset really {args.dataset}")
    
    prompter = Prompter("ko-wyvern")

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result['input_ids'][-1] != tokenizer.eos_token_id
            and len(result['input_ids']) < args.cutoff_len
            and add_eos_token
        ):
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)

        result['labels'] = result['input_ids'].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point['instruction'],
            data_point['prompt'],
            data_point['input'],
            data_point['response'],
        )

        tokenized_full_prompt = tokenize(full_prompt)
        
        return tokenized_full_prompt
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    if args.val_set_size > 0:
        train_val = dataset.train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val['train'].shuffle().map(generate_and_tokenize_prompt)
        )
        test_data = (
            train_val['test'].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = dataset.shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = Trainer(
        mdoel=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="epoch",
            eval_steps=200 if args.val_set_size > 0 else None,
            lr_scheduler_type=args.lr_scheduler,
            save_total_limit=1,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            group_by_length=args.group_by_length
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensor="pt", padding=True
        )
    )
    model.config.use_cache = False

    trainer.train()

    model.save_pretrianed(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)