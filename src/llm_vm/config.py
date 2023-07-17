import argparse, configparser
import re

parser = argparse.ArgumentParser()

# Set parse strategy here, lines should be self explanatory
parser.add_argument("-b", "--big_model", type=str, default='chat_gpt', help='Big LLM Model. Default: chat_gpt')
parser.add_argument("-c", "--config_file", type=str, help='Config File')
parser.add_argument("-p", "--port", type=int, default=3002, help='Port Number. Default: 3002')
parser.add_argument("-s", "--small_model", type=str, default='bloom', help='Small LLM Model. Default: gpt')
parser.add_argument("-H", "--host", type=str, default='127.0.0.1', help='Host Address. Default: 127.0.0.1')

args = parser.parse_args()

# Handling config file for customized launching
if args.config_file:
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("Defaults")))
    parser.set_defaults(**defaults)
    args = parser.parse_args() # Overwrite arguments

# argument checking

# making MODELS_AVAILABLE a set because it will be used for membership testing
MODELS_AVAILABLE = set([
    "opt",
    "bloom",
    "neo",
    "llama",
    "gpt",
    "chat_gpt",
])

if args.big_model not in MODELS_AVAILABLE:
    print(args.big_model + " is an invalid Model selection for Big LLM Model")
    exit()
    
if args.small_model not in MODELS_AVAILABLE:
    print(args.small_model + " is an invalid Model selection for Small LLM Model")
    exit()
    
# check args.port is a valid port number
if not (1024 < args.port < 65535):
    # we start at 1024 because port numbers below there are typical system processes
    # 65535 is 2**16-1 which represents the maximal value a port number can take
    print("The port number is out of the valid range (1024-65535). Reverting to the default port.")
    args.port = 3002

# check args.host is a valid IP address
pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
if not re.match(pattern, args.host):
    print("Invalid IP address. Reverting to the default host.")
    args.host = '127.0.0.1'
else:
    # validates each number in the IP address is between 0-255
    octets = args.host.split('.')
    valid_ip = True
    for octet in octets:
        if not 0 <= int(octet) <= 255:
            valid_ip = False
            break
    if not valid_ip:
        print("Invalid IP address range. Reverting to the default host.")
        args.host = '127.0.0.1'
