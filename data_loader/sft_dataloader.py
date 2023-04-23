#!python
# -*- coding: utf-8 -*-
# @author: Kun


from datasets import load_dataset


class SFTDataLoader(object):
    def __init__(self, data, CUTOFF_LEN, VAL_SET_SIZE, tokenizer) -> None:
        super(SFTDataLoader, self).__init__()

        self.data = data
        self.CUTOFF_LEN = CUTOFF_LEN
        self.VAL_SET_SIZE = VAL_SET_SIZE

        self.tokenizer = tokenizer

    def generate_prompt(self, data_point):
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}"""

    def tokenize(self, prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.CUTOFF_LEN + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    def generate_and_tokenize_prompt(self, data_point):
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        user_prompt = (
            (
                f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    """
            )
            if data_point["input"]
            else (
                f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    """
            )
        )
        len_user_prompt_tokens = (
            len(
                self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=self.CUTOFF_LEN + 1,
                )["input_ids"]
            )
            - 1
        )  # no eos token
        full_tokens = self.tokenizer(
            user_prompt + data_point["output"],
            truncation=True,
            max_length=self.CUTOFF_LEN + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def load_data(self):
        if self.VAL_SET_SIZE > 0:
            train_val = self.data["train"].train_test_split(
                test_size=self.VAL_SET_SIZE, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle().map(
                self.generate_and_tokenize_prompt)
            val_data = train_val["test"].shuffle().map(
                self.generate_and_tokenize_prompt)
        else:
            train_data = self.data["train"].shuffle().map(
                self.generate_and_tokenize_prompt)
            val_data = None

        return train_data, val_data
