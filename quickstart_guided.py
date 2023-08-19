#! /usr/bin/env python3
# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='pythia', small_model='neo')

# Put in your prompt and go!
response = client.complete(prompt = 'Is Barack Obama the first black president of the USA?',
                           context='',regex=r"\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)")
print(response)
response = client.complete(prompt = 'Did MJ win 6 titles with the Bulls',
                           context='',choices=["Hell Yeah Dude what a player","No way, Lebron's the Goat"])
print(response)
response = client.complete(prompt = 'How many presidents has the USA had?',
                           context='',type="integer")
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
