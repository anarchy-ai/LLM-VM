# import our client
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use
client = Client(big_model='llama')

# specify the file name of the finetuned model to load
model_name =  '2023-08-10T19:46:30_open_llama_3b_v2.pt'
client.load_finetune(model_name)

# Put in your prompt and go!
response = client.complete(prompt = '''
Subprompts:
Find all the planets in the solar system.,
Using previous find all the people on the 3rd planet.,
Using previous return all people that are christian.,
                           
Approved APIs:
                           
PlanetAPI: Find all the planets in the solar system.
Endpoint: https://planetapi.com/planets
PopulationAPI: Returns all the people on a planet
Endpoint: https://popapi.com/pop
CultureAPI: Filters a list of people by culture.
Endpoint: https://cultureapi.com/cult
JSON Schemas:
PlanetAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "object",
        "planet" :{
            "name": {"type":"string"}
        }
}
PopulationAPI:
Return:
{
    "type": "array",
    "items": {
        "type":"object",
        "person":{
             "type":"string"
       }
        
    }
}

CultureAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "object",
        "person": {
            "name": {"type": "string"}
        },
    }
}

Generate DryMerge code to answer the prompt. Use the subprompts:                                                  
                           ''',  context='')["completion"]
print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
