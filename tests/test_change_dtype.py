import sys
from llm_vm.client import Client

big_conf = {'dtype': 'float32'}
small_conf = {'dtype': 'float16'}

client = Client(
    big_model='pythia', 
    big_model_config=big_conf, 
    small_model_config=small_conf
)

response = client.complete(prompt = 'What is Anarchy?', context='')
print(response, file=sys.stderr)

client.change_model_dtype(big_model_dtype='float16')