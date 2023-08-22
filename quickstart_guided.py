#! /usr/bin/env python3
# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='pythia', small_model='neo')

# Put in your prompt and go!
response = client.complete(prompt = 'Is 1+1=10000?',
                           context='', openai_key=os.getenv("LLM_VM_OPENAI_API_KEY"), finetune = True, data_synthesis = True, regex=r"\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)")
print(response)
response = client.complete(prompt = 'Did MJ win 6 titles with the Bulls',
                           context='',choices=["Hell Yeah Dude what a player","No way, Lebron's the Goat"])
print(response)
response = client.complete(prompt = 'How many presidents has the USA had?',
                           context='',type="integer")
print(response)

