# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use
client = Client(big_model='pythia')

# specify the file name of the finetuned model to load
model_name =  '2023-08-08T22:44:34_pythia-70m-deduped.pt'
client.load_finetune(model_name)

# Put in your prompt and go!
response = client.complete(prompt = 'Q: I want to search the LLM-VM repository for all files that contain the token "bleeep".', context='')["completion"].split("<ENDOFLIST>")[0]
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
