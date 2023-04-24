#!python
# -*- coding: utf-8 -*-
# @author: Kun


from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers import LlamaForCausalLM, LlamaTokenizer

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable
    model_name: Optional[str] = field(default="./lora-alpaca_default_config", metadata={"help": "the model name"})
    # base_model_name: Optional[str] = field(default="decapoda-research/llama-7b-hf", metadata={"help": "the model name"}) # my code: this is not used.
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print("script_args: ", script_args)

peft_model_id = script_args.model_name
peft_config = PeftConfig.from_pretrained(peft_model_id)
print("peft_config: ", peft_config)

model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.float16,
    # ValueError: Loading THUDM/chatglm-6b requires you to execute the configuration file in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.
    # trust_remote_code=True,
)

# tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
# using above code, it will raise exception "ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported."
# reference  https://github.com/huggingface/transformers/issues/22222
# Hi @candowu, thanks for raising this issue. This is arising, because the tokenizer in the config on the hub points to LLaMATokenizer. However, the tokenizer in the library is LlamaTokenizer.
# This is likely due to the configuration files being created before the final PR was merged in.
tokenizer = LlamaTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
print("key_list: ", key_list)
for key in key_list:
    print("key: ", key)
    # peft==0.2.0 work
    parent, target, target_name = model.base_model._get_submodules(key) 
    # peft==0.3.0.dev0 class has no method _get_submodules, use code below, other error. WTF!
    # from peft.tuners.lora import _get_submodules
    # parent, target, target_name = _get_submodules(model.base_model, key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model

if script_args.output_name is None:
    output_name = f"{script_args.model_name}-adapter-merged"
    model.save_pretrained(output_name)
else:
    model.save_pretrained(f"{script_args.output_name}")
# model.push_to_hub(f"{script_args.model_name}-adapter-merged", use_temp_dir=False)