# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='chat_gpt', small_model='gpt')

# Put in your prompt and go!
response = client.complete(prompt = 'Is it warmer in Paris or Timbuktu and what are the temperatures in either city?',
                           context='',
                           openai_key=os.getenv("LLM_VM_OPENAI_API_KEY"), #for REBEL we need an OpenAI key
                           tools=
                           [{'description': 'Find the weather at a location and returns it in celcius.',
                            'dynamic_params': {"latitude": 'latitude of as a float',"longitude": 'the longitude as a float'},
                            'method': 'GET',
                            'url': "https://api.open-meteo.com/v1/forecast",
                            'static_params': {'current_weather': 'true'}}]) #No tools by default, so you have to add your own
print(response)

