# **Vicuna-LoRA-RLHF-PyTorch**
a full pipeline to finetune Vicuna LLM with LoRA and RLHF on consumer hardware


---
## **Table of Contents**
- [**Vicuna-LoRA-RLHF-PyTorch**](#vicuna-lora-rlhf-pytorch)
  - [**Table of Contents**](#table-of-contents)
  - [**Environment Setup**](#environment-setup)
  - [**Todo List**](#todo-list)
  - [**Run**](#run)
    - [**Download Vicuna Weights**](#download-vicuna-weights)
    - [**Supervised Finetune**](#supervised-finetune)
    - [**Merge PEFT adapter into Model**](#merge-peft-adapter-into-model)
    - [**Train Reward Model**](#train-reward-model)
      - [**Merge Reward adapter into Model**](#merge-reward-adapter-into-model)
    - [**Tuning LM with PPO**](#tuning-lm-with-ppo)
  - [**Topics**](#topics)
  - [**Reference**](#reference)
---

## **Environment Setup**
```
穷人卡：2080Ti 12G
torch==2.0.0
cuda==11.8
```

---
## **Todo List**
- [x] Download Vicuna Weights
- [x] SFT: Supervised Finetune
- [x] Merge Adapter into Model
- [x] RLHF
  - [x] train reward model
  - [x] tuning with RL

## **Run**
---

### **Download Vicuna Weights**

```bash
python apply_delta.py --base 'decapoda-research/llama-7b-hf' --target './weights/vicuna-7b' --delta lmsys/vicuna-7b-delta-v1.1
```

### **Supervised Finetune**

```
 check src/peft/utils/save_and_load.py , Only comment the line 52 to # #to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
```

```bash
python supervised_finetune.py --data_path './data/merge_sample.json' --output_path 'lora-Vicuna' --model_path './weights/vicuna-7b' --eval_steps 200 --save_steps 200 --test_size 1
```


### **Merge PEFT adapter into Model**

```bash
pip uninstall peft -y
pip install peft==0.2.0  # 0.3.0.dev0 has many errors

python merge_peft_adapter.py --model_name 'lora-Vicuna'

pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git # then comments peft/utis/save_and_load.py line 52.
```

### **Train Reward Model**

```
```

#### **Merge Reward adapter into Model**

```bash
```

### **Tuning LM with PPO**

```bash
```

---

## **Topics**
1. 第一步SFT之前，切记有个注意事项，需要检查下 安装的peft代码， src/peft/utils/save_and_load.py , 如果 line 52 有这行代码  #to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}，需要将其注释掉，否则在finetune完之后，保存不了 adapter model 的参数。切记！
2. PEFT的版本，目前从git上安装的是 0.3.0.dev0 版本，在merge_peft_adapter的时候有问题，需要切换到peft==0.2.0 (0.3.0.dev0 没有 _get_submodules()这个函数)
3. train reward model的时候 会发生另一个问题： ValueError: weight is on the meta device, we need a `value` to put in on 0. 需要参看 transformer 在github上的最新代码，我在发现这个问题的时候，隔天发现在transformer的github上 8小时前才刚刚修复了这个问题。
4. 最后一步，代码上基本是ok的，但是本人只有2080Ti的卡，加载完finetune model之后，再加载Reward model的时候 直接CUDA out of memory了，所以并未执行。


## **Reference**
utils & templates 来自 [alpaca-lora](https://github.com/tloen/alpaca-lora) 。

requirements 主要是按照 [alpaca-lora](https://github.com/tloen/
alpaca-lora) 来配环境。
* [https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
* [https://github.com/lvwerra/trl](https://github.com/lvwerra/trl)
* [https://github.com/jasonvanf/llama-trl](https://github.com/jasonvanf/llama-trl)