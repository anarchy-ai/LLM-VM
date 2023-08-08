#! /usr/bin/env python3
# import our client
from llm_vm.client import Client
import os
#from llm_vm.config import settings
# Instantiate the client specifying which LLM you want to use
client = Client(big_model='chat_gpt', small_model='llama')

# Put in your prompt and go!
response = client.complete(prompt = '''Prompt: "I want to know all the apartments in a 10-mile radius that cost less than 4000 a month."
                           
Subprompts:
Find all the apartments in your city.
Using the previous, find all that are 10 miles away.
Using the previous, find all that cost less than 4000 a month.
                           
Approved APIs:
                           
PropertyAPI: Fetches apartments based on city.
Endpoint: https://propertyapi.com/city
GeoAPI: Returns properties within a certain radius.
Endpoint: https://geoapi.com/radius
PriceFilterAPI: Filters properties based on price.
Endpoint: https://pricefilterapi.com/filter
JSON Schemas:
PropertyAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "address": {"type": "string"},
            "price": {"type": "number"},
            "coordinates": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                },
                "required": ["latitude", "longitude"]
            }
        },
        "required": ["id", "address", "price", "coordinates"]
    }
}
GeoAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "number"
    }
}
PriceFilterAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "number"}
        },
        "required": ["id"]
    }
}

Generate DryMerge code to answer the prompt. Use the subprompts:                                                  
                           ''', context='''
Prompt: "I want to know all the apartments in a 5-mile radius that cost less than 300 a month."
                           
Subprompts:
Find all the apartments in your city.
Using the previous, find all that are 5 miles away.
Using the previous, find all that cost less than 300 a month.
                           
Approved APIs:
                           
PropertyAPI: Fetches apartments based on city.
Endpoint: https://propertyapi.com/city
GeoAPI: Returns properties within a certain radius.
Endpoint: https://geoapi.com/radius
PriceFilterAPI: Filters properties based on price.
Endpoint: https://pricefilterapi.com/filter
JSON Schemas:
PropertyAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "address": {"type": "string"},
            "price": {"type": "number"},
            "coordinates": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"}
                },
                "required": ["latitude", "longitude"]
            }
        },
        "required": ["id", "address", "price", "coordinates"]
    }
}
GeoAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "number"
    }
}
PriceFilterAPI:
Return:
{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "number"}
        },
        "required": ["id"]
    }
}
                           
Generate Dry Merge code to answer prompt. Use the subprompts:
                           
[
    {
        "identity": {
            "namespace": "apartment_search",
            "name": "all_city_apartments"
        },
        "request_schema": {
            "core": {
                "url": "https://propertyapi.com/city",
                "method": "GET"
            },
            "query": {
                "city": {"dry_string": "{{context.city}}"}
            }
        }
    },
    {
        "identity": {
            "namespace": "apartment_search",
            "name": "within_radius"
        },
        "request_schema": {
            "core": {
                "url": "https://geoapi.com/radius",
                "method": "POST"
            },
            "body": {
                "apartment_ids": {"dry_value": "{{dependencies.all_city_apartments}}"},
                "radius": 5
            }
        },
        "dependency_schema": {
            "all_city_apartments": {
                "identity": "apartment_search/all_city_apartments"
            }
        }
    },
    {
        "identity": {
            "namespace": "apartment_search",
            "name": "price_filter"
        },
        "request_schema": {
            "core": {
                "url": "https://pricefilterapi.com/filter",
                "method": "POST"
            },
            "body": {
                "apartment_ids": {"dry_value": "{{dependencies.within_radius}}"},
                "max_price": 300
            }
        },
        "dependency_schema": {
            "within_radius": {
                "identity": "apartment_search/within_radius"
            }
        }
    }
    ]
                           ''',
                           openai_key = "",
                           temperature=0.0,
                           data_synthesis=True,
                           finetune=True,)
print(response)

# response = client.complete(prompt = "Answer question Q. ",context="Q: What is the economic situation in France",
#                            openai_key=settings.openai_api_key,
#                            temperature=0.0,
#                            data_synthesis=True,
#                            finetune=True,)
# print(response)
# response = client.complete(prompt = "Answer question Q. ",context="Q: What is the currency in myanmmar",
#                            openai_key=settings.openai_api_key,
#                            temperature=0.0,
#                            data_synthesis=True,
#                            finetune=True,)
# print(response)
# Anarchy is a political system in which the state is abolished and the people are free...
