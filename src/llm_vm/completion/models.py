import llm_vm.onsite_llm as llms
from llm_vm.config import args
import json
import os

print('Big LLM Model: ' + args.big_model)
print('Small LLM Model: ' + args.small_model)

# making MODELS_AVAILABLE a set because it will be used for membership testing
MODELS_AVAILABLE = set([
    "opt",
    "bloom",
    "neo",
    "llama",
    "gpt",
    "chat_gpt",
])

MODEL_DICT = {
    "opt":llms.Small_Local_OPT,
    "bloom":llms.Small_Local_Bloom,
    "neo":llms.Small_Local_Neo,
    "llama":llms.Small_Local_LLama,
    "gpt":llms.GPT3,
    "chat_gpt":llms.Chat_GPT
}

class ModelConfig:
    def __init__(self, big_model = 'chat_gpt', small_model='gpt'):
        self.big_model = MODEL_DICT[big_model]()
        self.small_model = MODEL_DICT[small_model]()

MODELCONFIG = ModelConfig(big_model=args.big_model, small_model=args.small_model)
