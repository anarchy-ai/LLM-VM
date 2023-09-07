#! /usr/bin/env python3
# import our client
import sys
from llm_vm.client import Client

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='opt',
                big_model_config={'model_uri': 'facebook/opt-350m'},
                small_model='opt',
                small_model_config={'model_uri': 'facebook/opt-125m'},
                optimizer='speculative_sampling',
                speculative_sampling_tokenizer="small",
                speculative_sampling_version="deepmind"
                )

# Put in your prompt and go!
response = client.complete(prompt = 'What is the capital of the USA?',
                           context='')
print(response, file=sys.stderr)
# Anarchy is a political system in which the state is abolished and the people are free...