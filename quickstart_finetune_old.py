#! /usr/bin/env python3
# import our client
from llm_vm.client import Client
import os
#from llm_vm.config import settings
# Instantiate the client specifying which LLM you want to use
client = Client(big_model='chat_gpt', small_model='llama')

# Put in your prompt and go!
response = client.complete(prompt = "Q: Locate and delete all emails that were sent to my manager last year.", 
                           context='Split the "Q" into its subtasks and return that as a list separated by commas. Return an empty string if no subtasks are necessary. \n\n Q: Find all the files in my system that were sent to HR before July 2nd. \n\n Find all files in system., Using previous answer search for files that were sent to HR., Using previous answer search for all files that were sent before July 2nd.<ENDOFLIST>',
                           openai_key="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",
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
