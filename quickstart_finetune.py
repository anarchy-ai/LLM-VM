#! /usr/bin/env python3
# import our client
from llm_vm.client import Client
import os
from llm_vm.config import settings
# Instantiate the client specifying which LLM you want to use
client = Client(big_model='chat_gpt', small_model='pythia')

# Put in your prompt and go!
response = client.complete(prompt = "Answer question Q. ",context="Q: What is the currency in myanmmar",
                           openai_key=settings.openai_api_key,
                           temperature=0.0,
                           data_synthesis=True,
                           finetune=True,)
print(response)

# response = client.complete(prompt = "Answer question Q. ",context="Q: What is the economic situation in France",
#                            openai_key=settings.openai_api_key,
#                            temperature=0.0,
#                            data_synthesis=True,
#                            finetune=True,)
# print(response)
# response = client.complete(prompt = "Answer question Q. ",context="Q: What is the currency in myanmmar",
#                            openai_key=settings.openai_api_key,
#                            temperature=0.0,
#                            data_synthesis=True,
#                            finetune=True,)
# print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
