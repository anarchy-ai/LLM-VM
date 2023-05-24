"""
This file has been temporarily repurposed as an interface that for users to 
call any of the three agents (REBEL, BACKWARD_CHAINING, and FLAT) and interact with them.
Running this file prompts the user to choose any of the agents and ask it questions. 
"""
import REBEL.agent 
import BACKWARD_CHAINING.agent
import keys  

def call_agent():
    print("Try out any agent!")

    # stores user input for which agent to try out 
    model_choice = 0 

    # times that the user was prompted to choose a valid model
    times_asked_for_model = 0 

    # if user enters invalid choice, prompt for input until valid 
    while True:
        model_choice = input("[1] FLAT (not working)\n[2] REBEL\n[3] BACKWARD_CHAINING\nChoose your agent:")
        try:
            # try to cast the input to an integer 
            model_choice = int(model_choice)

            if model_choice not in range (1, 4):
                print("=====Please enter 1, 2, or 3!=====")
            else:
                # user has entered a valid input 
                break
        except:
            print("=====Please enter 1, 2, or 3!=====")

    # FLAT 
    if model_choice == 1:
        # TODO: Add agent call here when FLAT is fixed
        pass
    # REBEL
    elif model_choice == 2:
        tools = REBEL.agent.buildGenericTools()
        agent = REBEL.agent.Agent(keys.OPENAI_DEFAULT_KEY, tools, verbose = 1)
    # BACKWARD_CHAINING 
    elif model_choice == 3:
        tools = BACKWARD_CHAINING.agent.buildGenericTools()
        agent = BACKWARD_CHAINING.agent.Agent(keys.OPENAI_DEFAULT_KEY, tools, verbose = 1)

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