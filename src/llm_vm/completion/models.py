import llm_vm.onsite_llm as llms
import json
import os

MODEL_DICT = {
    "opt":llms.Small_Local_OPT,
    "bloom":llms.Small_Local_Bloom,
    "neo":llms.Small_Local_Neo,
    "llama":llms.Small_Local_LLama,
    "gpt":llms.GPT3,
    "chat_gpt":llms.Chat_GPT
}

def return_generation_big():
    f = open('src/llm_vm/completion/config.json')
    data = json.load(f)
    return MODEL_DICT[data["big_model"]]

def return_generation_small():
    f = open('src/llm_vm/completion/config.json')
    data = json.load(f)
    return MODEL_DICT[data["small_model"]]