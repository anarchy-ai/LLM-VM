import os
import sys

# Get the current file's directory to grab the python files with common functionality in the utils/ folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# utils_dir = os.path.join(grandparent_dir, 'utils/')
# sys.path.append(utils_dir)

from keys import *
from labels import *
from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.agent_helper.utils import verbose_answer

CUSTOM_TOOL_ANSWER_EMBEDDING = "/answer_embedding"


def __get_generic_tools():
    # # wolfram

    WOLFRAM_KEY = os.getenv("LLM_VM_WOLFRAM_KEY")
    GOOGLE_MAPS_KEY = os.getenv("LLM_VM_GOOGLE_MAPS_KEY")
    wolfram_tool = {
        'description': "Useful to query questions about people, events, anything that can change, complicated math, live data retrieval, current date and other data.",
        # {'description': "The tool returns the results of free-form queries similar to those used for wolfram alpha. This is useful for complicated math or live data retrieval.  Can be used to get the current date.",
        'id': DefaultTools.WOLFRAM.value,
        'dynamic_params': {"input": 'The natural language input query'},
        'method': 'GET',
        'args': {
            'url': "http://api.wolframalpha.com/v2/query",
            'params': {'appid': WOLFRAM_KEY, 'input': '{input}'}
        },
    }

    # geopy
    directions_tool = {
        'description': "Find the driving distance and time to travel between two cities.",
        'id': DefaultTools.DIRECTIONS.value,
        'dynamic_params': {"origins": 'the origin city', "destinations": 'the destination city'},
        'method': 'GET',
        'args': {
            'url': "https://maps.googleapis.com/maps/api/distancematrix/json",
            'params': {
                'key': GOOGLE_MAPS_KEY,
                'origins': '{origins}',
                'destinations': '{destinations}'
            }
        }
    }

    # weather
    weather_tool = {
        'description': 'Useful to get the weather.',
        'ai_response_prompt': 'Returns temperature in celcius',
        'id': DefaultTools.WEATHER.value,
        'dynamic_params': {
            "latitude": 'latitude of the location, a float',
            "longitude": 'the longitude of the location, a float'
        },
        "ai_response_prompt": '''the value of "current_weather.weathercode" means:
0: Clear sky
1, 2, 3: Mainly clear, partly cloudy, and overcast
45, 48: Fog and depositing rime fog
51, 53, 55: Drizzle
56, 57: Freezing Drizzle
61, 63, 65: Rain
66, 67: Freezing Rain
71, 73, 75: Snow fall
77: Snow grains
80, 81, 82: Rain showers
85, 86: Snow showers slight and heavy
95: Thunderstorm
96, 99: Thunderstorm with slight and heavy hail

Do not return the weathercode as a number, instead use the description from list above''',
        'method': 'GET',
        'args': {
            'url': "https://api.open-meteo.com/v1/forecast",
            'params': {
                'current_weather': 'true',
                'latitude': '{latitude}',
                'longitude': '{longitude}'
            }
        }
    }

    # return [wolfram_tool, directions_tool, weather_tool]
    return [weather_tool]


GENERIC_TOOLS = __get_generic_tools()
