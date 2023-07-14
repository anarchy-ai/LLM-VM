import llm_vm.onsite_llm as llms
from llm_vm.config import args
import json
import os

# Dictionary of models to be loaded in ModelConfig
MODEL_DICT = {
    "opt":llms.Small_Local_OPT,
    "bloom":llms.Small_Local_Bloom,
    "neo":llms.Small_Local_Neo,
    "llama":llms.Small_Local_LLama,
    "gpt":llms.GPT3,
    "chat_gpt":llms.Chat_GPT
}

class ModelConfig:
    """
    This class loads specific big and small llm models based on intialization parameters

    Attributes:
        big_model (Base_Onsite_LLM): LLM model used as a reference source for fine-tuning
        small_model (Base_Onsite_LLM): LLM model used for fine-tuning off the big_model
    """
    def __init__(self, big_model = 'chat_gpt', small_model='gpt'):
        self.big_model = MODEL_DICT[big_model]()
        self.small_model = MODEL_DICT[small_model]()

# Loads the server with argument flags from command line
MODELCONFIG = ModelConfig(big_model=args.big_model, small_model=args.small_model)

