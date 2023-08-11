# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use
client = Client(big_model='llama')

# specify the file name of the finetuned model to load
model_name =  '2023-08-08T22:22:06_open_llama_3b_v2.pt'
client.load_finetune(model_name)

# Put in your prompt and go!
response = client.complete(prompt = 'Q: Considered the strongest recorded tropical cyclone, which cyclone had a film made about it in 2007?', context='')["completion"].split("<ENDOFLIST>")[0]
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
