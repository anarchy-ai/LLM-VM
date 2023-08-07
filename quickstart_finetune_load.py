# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use
client = Client(big_model='llama')

# specify the file name of the finetuned model to load
model_name = '2023-08-04T08:24:41_open_llama_3b_v2.pt'
client.load_finetune(model_name)

# Put in your prompt and go!
response = client.complete(prompt = 'Q: Tell me all the houses near me that cost less than 3000 a month by also are within 1 mile from a well reputed elementary school.', context='')["completion"]
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
