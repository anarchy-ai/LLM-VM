#! /usr/bin/env python3
# import our client
from llm_vm.client import Client
import os
#from llm_vm.config import settings
# Instantiate the client specifying which LLM you want to use
client = Client(big_model='chat_gpt', small_model='pythia')

# Put in your prompt and go!
response = client.complete(prompt = "Q: Which city in texas his furthest from Beijing?", 
                           context='Split the "Q" into its subtasks and return that as a list separated by commas. Return an empty string if no subtasks are necessary. \n\n Q: Which city has the highest temperature Beijing, Cabo, or Portland? \n\n Find the temperature of Beijing., Find the temperature of Cabo., Find the temperature of Portland., Using previos compare the three temperatures and return the highest.<ENDOFLIST>',
                           openai_key="",
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
