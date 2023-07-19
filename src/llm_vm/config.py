import re
import os
import argparse
from dynaconf import Dynaconf
from xdg import XDG_CONFIG_HOME


# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--big_model', type=str, help='Big LLM Model.')
parser.add_argument('-p', '--port', type=int, help='Port Number.')
parser.add_argument('-s', '--small_model', type=str, help='Small LLM Model.')
parser.add_argument('-H', '--host', type=str, help='Host Address.')
args = parser.parse_args()

# Set the CLI argument values to environment variables if they are present
if args.big_model is not None:
    os.environ['LLM_VM_BIG_MODEL'] = args.big_model
if args.port is not None:
    os.environ['LLM_VM_PORT'] = str(args.port)
if args.small_model is not None:
    os.environ['LLM_VM_SMALL_MODEL'] = args.small_model
if args.host is not None:
    os.environ['LLM_VM_HOST'] = args.host


source_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(source_directory, os.pardir, os.pardir))


print("Source Directory: " + source_directory)
print("Project Root: " + project_root)


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
MODELS_AVAILABLE = set([
    "opt",
    "bloom",
    "neo",
    "llama",
    "gpt",
    "chat_gpt",
])

if settings.big_model not in MODELS_AVAILABLE:
    print(settings.big_model + " is an invalid Model selection for Big LLM Model")
    exit()

if settings.small_model not in MODELS_AVAILABLE:
    print(settings.small_model + " is an invalid Model selection for Small LLM Model")
    exit()

# check args.host is a valid IP address
pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
if not re.match(pattern, settings.host):
    print("Invalid IP address. Reverting to the default host, 127.0.0.1")
    settings.host = '127.0.0.1'
else:
    # validates each number in the IP address is between 0-255
    octets = settings.host.split('.')
    valid_ip = True
    for octet in octets:
        if not 0 <= int(octet) <= 255:
            valid_ip = False
            break
    if not valid_ip:
        print("Invalid IP address range. Reverting to the default host, 127.0.0.1")
        settings.host = '127.0.0.1'


assert settings.big_model in MODELS_AVAILABLE, f"{settings.big_model} is not a valid Model selection for Big LLM Model"
assert settings.small_model in MODELS_AVAILABLE, f"{settings.small_model} is not a valid Model selection for Small LLM Model"
assert 1024 < settings.port < 65535, f"{settings.port} is an invalid port number"
