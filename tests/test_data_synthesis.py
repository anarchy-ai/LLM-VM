import sys
from dotenv import load_dotenv
import os
import openai
from llm_vm.completion.data_synthesis import DataSynthesis
from llm_vm.completion.optimize import *
from llm_vm.client import Client
import re

client = Client(big_model='gpt', small_model='neo')
if __name__ == "__main__":
    try:
        load_dotenv()
    except:
        pass
    openai.api_key = os.getenv('LLM_VM_OPENAI_API_KEY')
    print("key:", openai.api_key, file=sys.stderr)


    data_synthesizer = DataSynthesis(0.87, 10)

    # for one-shot prompt
    prompt = "What is the currency in myanmmar?"
    response = client.complete(prompt=prompt, context = "", openai_key="", temperature = 0.0)["completion"]
    f_response = re.sub('\n', '', response)
    print(f"Prompt: {prompt} \nResponse: {f_response}", file=sys.stderr)
    lst = data_synthesizer.data_synthesis(client.optimizer, prompt, response, openai_key="", temperature=0.0)
    print("SYNTHESIS COMPLETE:", lst)

    # for k-shot prompt
    # prompt_list = ["What is the currency in Madagascar?", "What is the currency in myanmmar?", "What is the currency in Morocco?"]
    # response_list = []
    # for p in prompt_list:
    #     res = client.complete(prompt=p, context = "", openai_key="", temperature = 0.0)
    #     response_list.append(res["completion"])

    # data_synthesizer.data_synthesis(client.optimizer, prompt_list, response_list, openai_key="", temperature=0.0)
