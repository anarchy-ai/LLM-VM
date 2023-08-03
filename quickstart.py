# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='bloom', small_model='pythia')

# Put in your prompt and go!
response = client.complete(prompt = 'Split Q into subquestions. Q: Which city is warmer today Timbuktu or Portland?',
                           context='')
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
