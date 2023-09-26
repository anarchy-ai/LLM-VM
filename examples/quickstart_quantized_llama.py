#! /usr/bin/env python3
# import our client
import sys
from llm_vm.client import Client
import os

# Instantiate the client specifying which LLM you want to use

client = Client(
    big_model='quantized-llama',
    small_model='quantized-llama',
    big_model_config={ 
        'model_uri': "TheBloke/LLaMa-7B-GGML",
        'model_kw_args': { 'model_file':'llama-7b.ggmlv3.q3_K_L.bin'}
      },
    small_model_config={ 
      'model_uri': "TheBloke/LLaMa-7B-GGML",
      'model_kw_args': { 'model_file':'llama-7b.ggmlv3.q3_K_L.bin'}
    },
  )

# Put in your prompt and go!
response = client.complete(prompt = 'What is the capital of the USA?',
                           context='')
print(response, file=sys.stderr)
# Anarchy is a political system in which the state is abolished and the people are free...