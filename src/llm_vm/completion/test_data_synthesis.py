from dotenv import load_dotenv
import os
import openai
import data_synthesis
from optimize import *


if __name__ == "__main__":
    try:
        load_dotenv()
    except:
        pass
    openai.api_key = os.getenv('OPENAI_KEY')
    print("key:", openai.api_key)

    data_synthesizer = data_synthesis.DataSynthesis(0,50)
    optimizer = LocalOptimizer(MIN_TRAIN_EXS=2)
    
    # for one-shot prompt
    prompt = "What is the currency in myanmmar?"
    response = CALL_BIG(prompt, temperature = 0.0)
    print(f"Prompt: {prompt} /nResponse: {response}")

    # for k-shot prompt
    prompt_list = ["What is the currency in Madagascar?", "What is the currency in myanmmar?", "What is the currency in Morocco?"]
    response_list = []
    for p in prompt_list:
        res = CALL_BIG(p, temperature = 0.0)
        response_list.append(res)
    print(f"Prompts: {prompt_list} /nResponses: {response_list}")


    data_synthesizer.data_synthesis(optimizer, prompt_list, response_list, temperature=0.0)