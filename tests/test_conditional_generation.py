from llm_vm.client import Client

# Select the LlaMA model
client = Client(small_model='bloom' , big_model='bloom')


response = client.complete(prompt="How many eyes does a spider have?" , context= "" , regex = "How many (teeth|ears) does a spider have")
print(response)


response = client.complete(prompt="Today's date is" , context= "" , regex = "([0-9]{2}/[0-9]{2}/[0-9]{4})")
print(response)
