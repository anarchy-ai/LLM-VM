"""
This file contains common tools (JSON API request objects) that are used
for all the agents to retrieve information for questions across various topics.
"""

import os
import sys

# Get the current file's directory to grab the python files with common functionality in the utils/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
utils_dir = os.path.join(grandparent_dir, 'utils/')
sys.path.append(utils_dir)

from llm_vm.utils.keys import *
from llm_vm.utils.labels import *
from llm_vm.utils.typings_llm import *

CUSTOM_TOOL_ANSWER_EMBEDDING = "/answer_embedding"


def __get_example_tools():
    # wolfram
    google_tool = {
               'description': "The tool returns the results of free-form queries similar to those used for wolfram alpha. This is useful for complicated math or live data retrieval.  Can be used to get the current date.",
               'dynamic_params': {"q": 'The natural language input query'},
               'method': 'GET',
               'args': {'url': "https://www.googleapis.com/customsearch/v1",
                         'params': {'key': "",
                                    'cx' : "",
                                    'q': '{q}'}
                        }
               }

    # geopy
    directions_tool = {'description': "Find the driving distance and time to travel between two cities.",
               'dynamic_params': {"origins": 'the origin city', "destinations": 'the destination city'},
               'method': 'GET',
               'args': {'url': "https://maps.googleapis.com/maps/api/distancematrix/json",
                         'params': {'key': "",
                                    'origins': '{origins}',
                                    'destinations': '{destinations}'}
                        }}

    # weather
    weather_tool = {'description': 'Find the weather at a location and returns it in celcius.',
               'dynamic_params': {"latitude": 'latitude of as a float',
                                  "longitude": 'the longitude as a float'},
               'method': 'GET',
               'args': {'url': "https://api.open-meteo.com/v1/forecast",
                         'params': {'current_weather': 'true',
                                    'latitude': '{latitude}',
                                    'longitude': '{longitude}'
                                    }}
               }
    return [google_tool, directions_tool, weather_tool]


GENERIC_TOOLS = __get_example_tools()
