#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2023-04-24 23:02:00
LastEditTime: 2023-04-24 23:02:03
LastEditors: Kun
Description: 
FilePath: /Vicuna-LoRA-RLHF-PyTorch/data_loader/rm_dataloader.py
'''


"""
data loader for reward modeling
"""

from datasets import load_dataset


class RewardDataLoader(object):
    def __init__(self, dataset_name, train_subset, eval_subset, num_proc, tokenizer) -> None:
        super(RewardDataLoader, self).__init__()

        self.dataset_name = dataset_name

        self.train_subset = train_subset
        self.eval_subset = eval_subset

        self.num_proc = num_proc
        self.tokenizer = tokenizer

    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
    # Then tokenize the dataset.

    def preprocess_function(self, examples):
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        # for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        for question, response_j, response_k in zip(examples["user_input"], examples["completion_a"], examples["completion_b"]):
            tokenized_j = self.tokenizer(
                "Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
            tokenized_k = self.tokenizer(
                "Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(
                tokenized_j["attention_mask"])
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(
                tokenized_k["attention_mask"])

        return new_examples

    def load_data(self):

        # Load the human stack-exchange-paired dataset for tuning the reward model.
        # train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/reward", split="train")
        train_dataset = load_dataset(self.dataset_name, split="train")
        if self.train_subset > 0:
            train_dataset = train_dataset.select(range(self.train_subset))
        # eval_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/evaluation", split="train")
        eval_dataset = load_dataset(self.dataset_name, split="train")
        if self.eval_subset > 0:
            eval_dataset = eval_dataset.select(range(self.eval_subset))

        original_columns = train_dataset.column_names

        # preprocess the dataset and filter out QAs that are longer than 512
        print("train_dataset: ", len(train_dataset))
        train_dataset = train_dataset.map(
            self.preprocess_function, batched=True, num_proc=self.num_proc, remove_columns=original_columns
        )
        train_dataset = train_dataset.filter(lambda x: len(
            x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512)
        print("train_dataset: ", len(train_dataset))

        print("eval_dataset: ", len(eval_dataset))
        eval_dataset = eval_dataset.map(
            self.preprocess_function, batched=True, num_proc=self.num_proc, remove_columns=original_columns)
        eval_dataset = eval_dataset.filter(lambda x: len(
            x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512)
        print("eval_dataset: ", len(eval_dataset))

        return train_dataset, eval_dataset
