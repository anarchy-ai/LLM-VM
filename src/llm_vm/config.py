import re
import os
import argparse
import sys
from dynaconf import Dynaconf
from xdg import XDG_CONFIG_HOME
from llm_vm.onsite_llm import model_keys_registered
from llm_vm.data_path import project_root
from ipaddress import ip_address


# project_root = os.path.abspath(os.getcwd())

print("Project Root: " + project_root, file=sys.stderr)


# default to the local file, and look in XDG if we can't find it
config_files = [
    os.path.join(project_root, "settings.default.toml"),
    os.path.join(XDG_CONFIG_HOME, "settings.toml"),
    os.path.join(project_root, "settings.toml"),
]

settings = Dynaconf(
    settings_file=config_files,
    load_dotenv=True,
    dotenv_path=".env",
    envvar_prefix="LLM_VM",
)

# making MODELS_AVAILABLE a set because it will be used for membership testing
MODELS_AVAILABLE = set(model_keys_registered)

if settings.big_model not in MODELS_AVAILABLE:
    print(settings.big_model + " is an invalid Model selection for Big LLM Model", file=sys.stderr)
    exit()

if settings.small_model not in MODELS_AVAILABLE:
    print(settings.small_model + " is an invalid Model selection for Small LLM Model", file=sys.stderr)
    exit()

# do we want to do this early test and exit?
# if settings.small_model is "chat_gpt":
#     print("openai currently doesn't support fine tuning chat_gpt, aka gpt3.5-turbo, exiting")
#     exit()

def isOpenAIModel(str):
    return str =="gpt" or str =="chat_gpt"

if settings.openai_api_key is None and (isOpenAIModel(settings.small_model) or isOpenAIModel(settings.big_model)):
    print("Error: you must have an OpenAI API key set via config files, ./settings.default.toml or via environment variable ", file=sys.stderr)
    print("LLM_VM_OPEN_AI_API,if you wish to use their models. Exiting", file=sys.stderr)
    exit()

# check settings.host is a valid IP address
def isValidIP(ip):
    try:
        # if object created IP is valid
        tmp = ip_address(ip)
        return True
    except ValueError:
        return False

if isValidIP(settings.host) is False:
    print("Invalid IP4/IP6 address. Reverting to the default host: 127.0.0.1", file=sys.stderr)
    settings.host = '127.0.0.1'



assert settings.big_model in MODELS_AVAILABLE, f"{settings.big_model} is not a valid Model selection for Big LLM Model"
assert settings.small_model in MODELS_AVAILABLE, f"{settings.small_model} is not a valid Model selection for Small LLM Model"
assert 1024 < settings.port < 65535, f"{settings.port} is an invalid port number"
