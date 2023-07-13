import argparse, configparser


parser = argparse.ArgumentParser()

# Set parse strategy here, lines should be self explanatory
parser.add_argument("-b", "--big_model", type=str, default='chat_gpt', help='Big LLM Model. Default: chat_gpt')
parser.add_argument("-c", "--config_file", type=str, help='Config File')
parser.add_argument("-p", "--port", type=int, default=3002, help='Port Number. Default: 3002')
parser.add_argument("-s", "--small_model", type=str, default='gpt', help='Small LLM Model. Default: gpt')
parser.add_argument("-H", "--host", type=str, default='127.0.0.1', help='Host Address. Default: 127.0.0.1')

args = parser.parse_args()

if args.config_file:
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("Defaults")))
    parser.set_defaults(**defaults)
    args = parser.parse_args() # Overwrite arguments

# at this point args can be imported into the rest of the project through llm_vm.server.anarchparse
