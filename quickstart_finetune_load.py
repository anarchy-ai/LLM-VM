# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use
client = Client(big_model='llama')

# specify the file name of the finetuned model to load
model_name = '2023-08-03T20:29:27_open_llama_3b_v2.pt'
client.load_finetune(model_name)

# Put in your prompt and go!
response = client.complete(prompt = 'Split Q into subquestions. Q: Which city is warmer beijing or shanghai?',
                           context='')
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
