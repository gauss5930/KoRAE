import os
from typing import List
import huggingface_hub

import fire
import torch
from datasets import load_dataset, Dataset

from trl import SFTTrainer
from accelerate import Accelerator

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from utils.prompter import Prompter

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

def create_datasets(data_path, val_set_size):
    dataset = load_dataset(
        data_path,
        split="train",
    )

    if val_set_size > 0:
        train_val = dataset.train_test_split(test_size=val_set_size, seed=42)

        train_data = train_val["train"]
        val_data = train_val["test"]
    else:
        train_data = dataset
        val_data = None

    return train_data, val_data

def train(
    # model/data params
    model_path: str = "beomi/llama-2-koen-13b", 
    data_path: str = "Cartinoe5930/KoRAE_filtered_12k",
    output_dir: str = "",
    hf_token: str = "",
    hf_hub_path: str = "",
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    gradient_checkpointing: bool = True,
    learning_rate: float = 1e-5,
    seq_len: int = 4096,
    val_set_size: int = 0,
    logging_steps: int = 1,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.03,
    packing: bool = False,
    save_strategy: str = "steps",
    save_steps: int = 100,
    evaluation_strategy: str = "steps",
    eval_steps: int = 100,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    prompt_template_name: str = "KoRAE_template"
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"model_path: {model_path}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"hf_token: {hf_token}\n"
            f"hf_hub_path: {hf_hub_path}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"gradient_checkpointing: {gradient_checkpointing}\n"
            f"learning_rate: {learning_rate}\n"
            f"seq_len: {seq_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"logging_steps: {logging_steps}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"packing: {packing}\n"
            f"save_strategy: {save_strategy}\n"
            f"save_steps: {save_steps}\n"
            f"evaluation_strategy: {evaluation_strategy}\n"
            f"eval_steps: {eval_steps}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
        )
    assert (
        model_path
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    huggingface_hub.login(hf_token)
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        use_auth_token=hf_token)
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_auth_token=hf_token
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, eval_dataset = create_datasets(data_path=data_path, val_set_size=val_set_size)

    train_dataset = process_dataset(train_dataset)
    eval_dataset = process_dataset(eval_dataset) if eval_dataset else None
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size if eval_dataset else None,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        evaluation_strategy=evaluation_strategy if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        bf16=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )
    model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="prompted_input",
        data_collator=data_collator,
        packing=packing,
        max_seq_length=seq_len,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model.push_to_hub(hf_hub_path)
    tokenizer.push_to_hub(hf_hub_path)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(train)