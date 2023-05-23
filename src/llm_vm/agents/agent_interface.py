from REBEL.agent import * 
from BACKWARD_CHAINING.goal_chatgpt_simple_agent import * 

def call_agent():
    print("Try out any agent!")

    
    # stores user input for which agent to try out 
    model_choice = 0 

    # if user enters invalid choice, prompt for input until valid 
    while model_choice not in range (1, 4):
        model_choice = input("[1] FLAT (not working)\n[2] REBEL\n[3] BACKWARD_CHAINING\nChoose your agent:")
        try:
            model_choice = int(model_choice)
        except:
            print("=====Please enter a valid input!=====")

    # FLAT 
    if model_choice == 1:
        pass
    # REBEL
    elif model_choice == 2:
        tools = buildGenericTools()
        agent = Agent(OPENAI_DEFAULT_KEY, tools, verbose = 1)
    # BACKWARD_CHAINING 
    elif model_choice == 3:
        tools = buildGenericTools()
        agent = Agent(OPENAI_DEFAULT_KEY, tools, verbose = 1)

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