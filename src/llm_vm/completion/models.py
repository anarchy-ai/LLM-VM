import llm_vm.onsite_llm as llms
from llm_vm.server.anarchparse import args
import json
import os

print(args.big_model)
print(args.small_model)

MODEL_DICT = {
    "opt":llms.Small_Local_OPT,
    "bloom":llms.Small_Local_Bloom,
    "neo":llms.Small_Local_Neo,
    "llama":llms.Small_Local_LLama,
    "gpt":llms.GPT3,
    "chat_gpt":llms.Chat_GPT
}

class ModelConfig:
    def __init__(self):
        f = open('src/llm_vm/completion/config.json')
        data = json.load(f)
        self.big_model = MODEL_DICT[args.big_model]()
        self.small_model = MODEL_DICT[args.small_model]()

MODELCONFIG = ModelConfig()
