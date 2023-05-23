"""
API Key Definitions and Regex Pattern for Pure Interpolations.

This section defines and retrieves the API keys required for various services from the environment variables. 
It also includes a regular expression pattern used to identify pure interpolations.

API Key Definitions:
- OPENAI_DEFAULT_KEY: The API key for OPENAI.
- GOOGLE_MAPS_KEY: The API key for Google Maps.
- SERPAPI_KEY: The API key for SERPAPI (Anarchy AI)
- WOLFRAM_KEY: The API key for Wolfram. 


Note: The corresponding environment variables should be set before running 
any of the agents. Set the environment variables without quotes. 
E.g.: WOLFRAM_KEY=an-encypted-key 
"""
import os
import openai

OPENAI_DEFAULT_KEY =os.environ["OPENAI_API_KEY"]
GOOGLE_MAPS_KEY = os.environ["GOOGLE_MAPS_KEY"]
SERPAPI_KEY = os.environ["SERPAPI_KEY"]
WOLFRAM_KEY = os.environ["WOLFRAM_KEY"]

###
# Use this regex to find if a variable is a pure interpolation.
# For @example, { "name": "{my_name}" } is pure, since there's nothing else besides the interpolation
# For @example, { "Ticker": "TickerID={ticker_id}|historical=1" } is NOT pure, since there is extra text
DICT_KEY_REGEX_TO_FIND_PURE_INTERPOLATIONS = "^\{[a-zA-Z0-9\.\-_]+\}$"

def set_openai_key(key=OPENAI_DEFAULT_KEY):
    os.environ["OPENAI_API_KEY"]=key
    openai.api_key=str(key)