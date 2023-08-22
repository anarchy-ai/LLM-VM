import os
import sys
import re
import random
from llm_vm.agents.FLAT.agent_helper.business_logic import promptf
from llm_vm.utils.labels import *
from llm_vm.agents.FLAT.agent_helper.utils import *
from llm_vm.agents.FLAT.agent_helper.tools import *
from llm_vm.utils.keys import *


random_fixed_seed = random.Random(4)

class Agent:
    def __init__(self, openai_key, tools = None, bot_instructions = "", verbose = 4):
        self.verbose = verbose
        self.set_tools((GENERIC_TOOLS + tools) if tools else GENERIC_TOOLS)
        self.bot_instructions = f"<{L_BOT_INSTRUCTIONS}>{bot_instructions}<{L_BOT_INSTRUCTIONS}>" if bot_instructions else ""

        # set the openai key to make calls to the API
        set_api_key(openai_key)
    def set_tools(self, tools):
        self.tools = []
        for tool in tools:
            if not 'args' in tool:
                tool['args'] = {}
            if not 'method' in tool:
                tool['method'] = "GET"
            if not 'examples' in tool:
                tool['examples'] = []
            if not 'dynamic_params' in tool:
                tool['dynamic_params'] = {}
            if not 'id' in tool:
                tool['id'] = len(self.tools) + 1

            self.tools += [tool]


    def run(self, question: str, memory: TupleList) -> Tuple[str, TupleList, list, DebugCallList]:
        try:
            answer, _, calls, debug_return, price = promptf(
                question,
                memory,
                self.tools,
                self.bot_instructions,
                self.verbose
            )

            # Remove xml tags from answer, if any. Important, because this is trade secret.
            answer, has_friendly_tags = remove_tags_from_html_string(answer)

        except Exception as e:
            print(e, flush=True, file=sys.stderr)
            print("Main thread exception: ", e, flush=True, file=sys.stderr)
            answer, calls, debug_return, price, has_friendly_tags = "Error: " + str(e), [], [], 0, False


        if self.verbose > -1:
            print_big("GPT-3.5 Price = ~{:.1f} cents".format(price * 100))

        return (answer, memory + [(question, answer)], calls, debug_return, has_friendly_tags)

def flat_main():
    # tools = [{'method': 'GET',"description":"use this tool to find the price of stocks",'args' : {"url":"https://finnhub.io/api/v1/quote",'params': { 'token' :'cfi1v29r01qq9nt1nu4gcfi1v29r01qq9nt1nu50'} },"dynamic_params":{"symbol":"the symbol of the stock"}}]
    tools =  [{'method': 'GET', "dynamic_params": { 'location': 'This string indicates the geographic area to be used when searching for businesses. \
    Examples: "New York City", "NYC", "350 5th Ave, New York, NY 10118".', 'term': 'Search term, e.g. "food" or "restaurants". The \
    term may also be the business\'s name, such as "Starbucks"', 'price': 'Pricing levels to filter the search result with: 1 = \
    $, 2 = $$, 3 = $$$, 4 = $$$$. The price filter can be a list of comma delimited pricing levels. e.g., "1, 2, 3" will filter the \
    results to show the ones that are $, $$, or $$$.'}, "description":"This tool searches for a business on yelp.  It's useful for finding restaurants and \
    whatnot.", 'args' :{'url': 'https://api.yelp.com/v3/businesses/search', 'cert': '', 'json': {}, 'params': {'limit': '1',
                                                                                                              'open_now': 'true', 'location': '{location}', 'term': '{term}', 'price': '{price}'}, 'data': {},
                       'headers': {'authorization': 'Bearer OaEqVSw9OV6llVnvh9IJo92ZCnseQ9tftnUUVwjYXTNzPxDjxRafYkz99oJKI9WHEwUYkiwULXjoBcLJm7JhHj479Xqv6C0lKVXS7N91ni-nRWpGomaPkZ6Z1T0GZHYx',
                                   'accept': 'application/json'}}}]

    label = Agent(os.getenv("LLM_VM_OPENAI_API_KEY"), tools, verbose=4)
    conversation_history = []
    last = ""
    while True:
        inp = input(last+"Human: ")
        return_value = label.run(inp, conversation_history)
        conversation_history = return_value[1]
        print(return_value[2], file=sys.stderr)
        last = "AI: "+str(return_value[0]) + "\n"

# print_op(google(' {"question": ""}'))
if __name__ == "__main__":
    flat_main()
