"""
This file has been temporarily repurposed as an interface for users to
call any of the three agents (REBEL, BACKWARD_CHAINING, and FLAT) and interact with them.
Running this file prompts the user to choose any of the agents and ask it questions.
"""
import sys
import llm_vm.agents.REBEL.agent as REBEL
import llm_vm.agents.FLAT.agent as FLAT
import os
key = os.getenv("LLM_VM_OPENAI_API_KEY")

def call_agent():
    print("Try out any agent!", file=sys.stderr)

    # stores user input for which agent to try out
    model_choice = 0

    # times that the user was prompted to choose a valid model
    times_asked_for_model = 0

    # if user enters invalid choice, prompt for input until valid
    while True:
        model_choice = input("[1] FLAT\n[2] REBEL\nChoose your agent:")
        try:
            # try to cast the input to an integer
            model_choice = int(model_choice)

            if model_choice not in range (1, 4):
                print("=====Please enter 1, or 2!=====", file=sys.stderr)
            else:
                # user has entered a valid input
                break
        except:
            print("=====Please enter 1 or 2!=====", file=sys.stderr)

    # FLAT
    if model_choice == 1:
        # TODO: Add agent call here when FLAT is fixed
        tools =  [{'method': 'GET', "dynamic_params": { 'location': 'This string indicates the geographic area to be used when searching for businesses. \
    Examples: "New York City", "NYC", "350 5th Ave, New York, NY 10118".', 'term': 'Search term, e.g. "food" or "restaurants". The \
    term may also be the business\'s name, such as "Starbucks"', 'price': 'Pricing levels to filter the search result with: 1 = \
    $, 2 = $$, 3 = $$$, 4 = $$$$. The price filter can be a list of comma delimited pricing levels. e.g., "1, 2, 3" will filter the \
    results to show the ones that are $, $$, or $$$.'}, "description":"This tool searches for a business on yelp.  It's useful for finding restaurants and \
    whatnot.", 'args' :{'url': 'https://api.yelp.com/v3/businesses/search', 'cert': '', 'json': {}, 'params': {'limit': '1',
                                                                                                              'open_now': 'true', 'location': '{location}', 'term': '{term}', 'price': '{price}'}, 'data': {},
                       'headers': {'authorization': '',
                                   'accept': 'application/json'}}}]
        agent = FLAT.Agent(key, tools, verbose=1)
    elif model_choice == 2:
        tools = REBEL.buildExampleTools()
        agent = REBEL.Agent(key, tools, verbose = 1)


    pass

    mem = []
    last = ""
    while True:
        inp = input(last+"Human: ")
        ret = agent.run(inp, mem)
        mem = ret[1]
        last = "AI: "+str(ret[0])+ "\n"

if __name__ == "__main__":
    call_agent()
