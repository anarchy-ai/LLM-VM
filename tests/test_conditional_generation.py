from llm_vm.client import Client

# Select the LlaMA model
client = Client(small_model='neo' , big_model='neo')

# Put in your prompt and go!
# response = client.complete(prompt = 'Is this a good demo?', context = '' , regex = "(Yes|No)")
# print(response)

response = client.complete(prompt = 'How many eyes does a spider have?' , context = '' ,  regex = "Spiders have [0-9] eyes")
print(response)
# Anarchy is a political philosophy that advocates no government...