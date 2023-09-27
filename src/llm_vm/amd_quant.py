from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# for AMD run "pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm542/"
import torch

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

# example input IDs and attention mask to use in quantizing model weights
examples = [
    tokenizer(
        "You can write any text here, it serves as an input for the quantization process."
    )
]

print("EXAMPLES: ", examples)

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,  
    desc_act=False,  
)

print(torch.has_cuda)

# load un-quantized model, by default, the model will always be loaded into CPU memory
# model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# print(model.device)

# model.quantize(examples)

# print(model.device)

# model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

# model.generate()
