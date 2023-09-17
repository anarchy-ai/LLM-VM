#! /usr/bin/env python3
# import our client

from llm_vm.client import Client
from llm_vm.config import settings

# Instantiate the client specifying which LLM you want to use

client = Client(big_model='pythia', small_model='pythia')

# Put in your prompt and go!
# response = client.complete(
#     prompt='Is 1+1=10000?',
#     context='',
#     openai_key=settings.openai_api_key,
#     finetune=False,
#     data_synthesis=False,
#     regex=r"\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)",
# )
# print(response)
# response = client.complete(
#     prompt='Did MJ win 6 titles with the Bulls',
#     context='',
#     openai_key=settings.openai_api_key,
#     finetune=False,
#     data_synthesis=False,
#     choices=["Hell yeah dude that's correct", "No way, that's hella false"],
# )
# print(response)
response = client.complete(
    prompt='What does 1+1 equal?', context='', openai_key=settings.openai_api_key, finetune=True, data_synthesis=True, type="integer"
)
print(response)
