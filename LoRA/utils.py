import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_name_or_path = "google/gemma-2b"  

# Q_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    cache_dir="./model_cache",
    # device_map='auto',
    # quantization_config=Q_config,
    #torch_dtype=torch.float16,
)
# model.eval()

tokenized = lambda txt: tokenizer.encode( 
    txt, 
    return_tensors="pt", 
    add_special_tokens=False
).to('cuda')
untokenize = lambda ids: tokenizer.batch_decode(
    ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import torch.nn as nn
import loralib as lora

def convert_to_lora(model, r=8, lora_alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 用 LoRALinear 替换 nn.Linear
            lora_module = lora.trueSVDLinear(module.in_features, module.out_features, r=r, lora_alpha=lora_alpha)
            lora_module.weight = module.weight
            if module.bias is not None:
                lora_module.bias = module.bias
            # 用 LoRA 版本的线性层替换原始的线性层
            parent = model
            *path, last = name.split('.')
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, lora_module)

convert_to_lora(model, r=8, lora_alpha=4)
lora.mark_only_lora_as_trainable(model)

model.to('cuda')

# print(lora.lora_state_dict(model))




#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def soft_p_list(in_seq_ids, t):
    torch.cuda.empty_cache()
    with torch.no_grad():
        logits = model( in_seq_ids )["logits"][0][-1].detach()
        
    return torch.softmax( logits/t, dim=0 ).to("cpu")

def greedy_gen(in_seq_ids, only_output=False): 
    torch.cuda.empty_cache()
    with torch.no_grad():
        full_seq = model.generate( 
            input_ids = in_seq_ids,
            #top_k = 1,
            do_sample = False,
            # pad_token_id = tokenizer.eos_token_id, 
        ) 
    return [ it[ len(in_seq_ids[0]) : -1 ] for it in full_seq ] if only_output else full_seq