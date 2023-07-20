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



###
# Use this regex to find if a variable is a pure interpolation.
# For @example, { "name": "{my_name}" } is pure, since there's nothing else besides the interpolation
# For @example, { "Ticker": "TickerID={ticker_id}|historical=1" } is NOT pure, since there is extra text
DICT_KEY_REGEX_TO_FIND_PURE_INTERPOLATIONS = "^\{[a-zA-Z0-9\.\-_]+\}$"


def set_api_key(key, key_type="OPENAI_API_KEY"):
    """
    Set the API key in the environment variable.

    Parameters:
    - key:  The API key to set. Defaults to OPENAI_DEFAULT_KEY.
            There are four options:
                - OPENAI_DEFAULT_KEY
                - GOOGLE_MAPS_KEY
                - GOOGLE_KEY
                - GOOGLE_CX
    - key_type: The type of API key. Defaults to "OPENAI_API_KEY".
                There are four options:
                - OPENAI_API_KEY
                - GOOGLE_MAPS_KEY
                - GOOGLE_KEY
                - GOOGLE_CX
    """
    if not os.getenv(key_type):
        os.environ[key_type]=key

        if key_type == "OPENAI_API_KEY":
            openai.api_key=str(key)
