# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='pythia')
print('Default load result')
response = client.complete(prompt = 'what is anarchy?',
                           context = '',)
print(response)
path_to_model = '~/.llm_vm/models/finetuned_models/pythia-70m-deduped/'
model_name = '2023-07-19T22:42:35_pythia-70m-deduped.pt'
client.load_finetune(model_name)
print('Final Trial')
# Put in your prompt and go!
response2 = client.complete(prompt = 'What is anarchy?',
                           context='')
print(response2)
# Anarchy is a political system in which the state is abolished and the people are free...
