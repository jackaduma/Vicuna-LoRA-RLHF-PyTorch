#!python
# -*- coding: utf-8 -*-
# @author: Kun

import os
import sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
import argparse
import warnings
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from data_loader import sft_dataloader

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="lora-Vicuna")
parser.add_argument("--model_path", type=str,
                    default="decapoda-research/llama-7b-hf")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--test_size", type=int, default=200)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
args = parser.parse_args()

if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"
# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't always need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.test_size  # 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
# "/home/cciip/private/fanchenghao/dataset/instruction/merge.json"
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_path  # "lora-Vicuna"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
print(args.model_path)
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained(
    args.model_path, add_eos_token=True
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
# tokenizer.padding_side = "left"  # Allow batched inference

data = load_dataset("json", data_files=DATA_PATH)

now_max_steps = max(
    (len(data["train"]) - VAL_SET_SIZE) // BATCH_SIZE * EPOCHS, EPOCHS)
if args.resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn(
                "The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = (
                None  # So the trainer won't try loading its state
            )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")

    train_args_path = os.path.join(
        args.resume_from_checkpoint, "trainer_state.json")

    if os.path.exists(train_args_path):
        import json
        base_train_args = json.load(open(train_args_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            warnings.warn("epoch {} replace to the base_max_steps {}".format(
                EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
else:
    MAX_STEPS = now_max_steps


model.print_trainable_parameters()


dataloader = sft_dataloader.SFTDataLoader(
    data, CUTOFF_LEN, VAL_SET_SIZE, tokenizer)
train_data, val_data = dataloader.load_data()

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=30,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False)
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

print("\n If there's a warning about missing keys above, please disregard :)")

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)
