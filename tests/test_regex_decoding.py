from llm_vm.onsite_llm import *
from transformers import AutoTokenizer
import torch

if __name__ == "__main__":
    model = HFTransformersWithRegex("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", add_prefix_space=True)

    input_text = "The one performing the heart surgery is a"
    input_ids = tokenizer(input_text, return_tensors="pt")
    model_input = input_ids["input_ids"]

    r_string = r"doctor|person"
    r_bias = 5.0

    class DebugStreamer:
        def __call__(self, tokens, scores):
            print(tokenizer.decode(tokens[-1]))
            return False
        
    output = model.generate(model_input, torch.ones_like(model_input), 1.0, 10, DebugStreamer(), r_string, r_bias)
    print("Generated output: ", tokenizer.batch_decode(output[0], skip_special_tokens=True))