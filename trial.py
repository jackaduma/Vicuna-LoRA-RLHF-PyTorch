from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
import bitsandbytes
import torch 
import gc
torch.cuda.empty_cache()

base_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
base = LlamaForCausalLM.from_pretrained(
        base_model_path, load_in_8bit=True,device_map="cuda"
    )
base_tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)
print("done")