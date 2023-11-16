#! /usr/bin/env python3
# import our client
import sys
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='gpt', small_model='gpt')

# Put in your prompt and go!
response = client.complete(prompt = 'What is the capital of the USA?',
                           context='')
print(response, file=sys.stderr)
# Anarchy is a political system in which the state is abolished and the people are free...