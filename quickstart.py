# import our client
from llm_vm.client import Client
import os
# Instantiate the client specifying which LLM you want to use
client = Client(big_model='gpt', small_model='neo')

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?',
                           context='',
                           openai_key=os.getenv("OPENAI_API_KEY"))
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
