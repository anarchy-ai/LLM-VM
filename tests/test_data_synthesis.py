from dotenv import load_dotenv
import os
import openai
import data_synthesis
from optimize import *
from llm_vm.client import Client

client = Client(big_model='gpt', small_model='neo')
if __name__ == "__main__":
    try:
        load_dotenv()
    except:
        pass
<<<<<<< HEAD:src/llm_vm/completion/test_data_synthesis.py
    openai.api_key = os.getenv('LLM_VM_OPENAI_API_KEY')
    print("key:", openai.api_key)
=======
>>>>>>> main:tests/test_data_synthesis.py

 
    data_synthesizer = data_synthesis.DataSynthesis(0.87, 50)
<<<<<<< HEAD:src/llm_vm/completion/test_data_synthesis.py
    optimizer = LocalOptimizer(MIN_TRAIN_EXS=2)

=======
    
>>>>>>> main:tests/test_data_synthesis.py
    # for one-shot prompt
    prompt = "What is the currency in myanmmar?"
    response = client.complete(prompt=prompt, context = "", openai_key="",temperature = 0.0)["completion"]
    print(f"Prompt: {prompt} /nResponse: {response}")
    
    # for k-shot prompt
    prompt_list = ["What is the currency in Madagascar?", "What is the currency in myanmmar?", "What is the currency in Morocco?"]
    response_list = []
    for p in prompt_list:
        res = client.complete(prompt=p, context = "", openai_key="", temperature = 0.0)
        response_list.append(res["completion"])
    



    data_synthesizer.data_synthesis(client.optimizer, prompt_list, response_list,openai_key="", temperature=0.0)
