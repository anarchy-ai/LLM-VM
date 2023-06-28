"""
This Python script runs the BACKWARD_CHAINING agent, which takes into account of stable context
and dynamic parameters to feed into the LLM model for more accurate results.  
The script also takes into account of previous prompts in its memory as context for questions the 
user may input later. 
The user is prompted to enter a question, to which the AI responds. 
The user will also see messages regarding which API for information was chosen and the 
price of the API call.  
"""

# run standalone like "python3 -m matt_agent.goal_chatgpt_simple_agent"
from operator import index
import os
import sys
from urllib.parse import urlencode
import urllib.parse as urlparse
import openai
import json
from llama_index import (
    Document,
    LLMPredictor,
    GPTVectorStoreIndex,
    PromptHelper,
    GPTTreeIndex,
)
import requests

# Get the current file's directory to grab the python files with common functionality in the utils/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(grandparent_dir, "utils/")
sys.path.append(utils_dir)

from keys import *
from print_types import *
from labels import *
from tools import *
from typings import *

try:
    from .utils import *
    from .tool_picker import *
    from .disambiguate import *
    from .extract import *
    from .contained import *
except:
    from utils import *
    from tool_picker import *
    from disambiguate import *
    from extract import *
    from contained import *


def buildGenericTools(tools=GENERIC_TOOLS):
    # wolfram (math & live facts)
    tools[0]["examples"] = [
        ("How many people are in Germany?"),
        ("What is the date today?"),
        ("Area of the place."),
        ("What is the date multiplied by seventy?"),
        ("Is 5073 raised to the 3rd power divisible by 73?"),
    ]

    # geopy
    tools[1]["examples"] = [
        [
            (
                "I'm in Paris and am curious about how long it would take to get to Zurich."
            ),
            ("How long would it take to get between South Africa and Kenya."),
            (
                "How many miles would the elephants travel going from Alaska to Montreal?"
            ),
        ]
    ]
    # weather
    tools[2]["examples"] = [
        ("What is the weather in NYC?"),
        ("What's Milan's wind-speed?"),
    ]
    return tools


class Agent:
    def __init__(self, openai_key, tools, verbose=4):
        self.verbose = verbose

        self.set_tools(tools)

        # set all the API resource keys to make calls
        set_api_key(openai_key, "OPENAI_API_KEY")
        set_api_key(GOOGLE_MAPS_KEY, "GOOGLE_MAPS_KEY")
        set_api_key(SERPAPI_KEY, "SERPAPI_KEY")
        set_api_key(WOLFRAM_KEY, "WOLFRAM_KEY")

    def makeToolDesc(self, tool_id):
        """
        Creates a tool description that contains the relevant tags according to the dynamic params

        Parameters
        ----------
        tool_id
            the tool's enum value as specified in the DefaultTools class

        Returns
        ----------
        String
            a formatted string with tags containing the tool_id, description and params
        """

        tool = self.tools[tool_id]
        params = (
            "{"
            + ", ".join(
                ['"' + l + '": ' + v for l, v in tool["dynamic_params"].items()]
            )
            + "}"
            if tool["dynamic_params"] != ""
            else "{}"
        )
        return (
            "<TOOL>"
            + f"<{TOOL_ID}>{str(tool_id)}</{TOOL_ID}>"
            + f"<{DESCRIPTION}>{tool['description']}</{DESCRIPTION}>"
            + f"<{PARAMS}>{params}</{PARAMS}>"
            + "</TOOL>"
        )

    def set_tools(self, tools):
        """
        Adds all the available tools when the class is initialized.

        Parameters
        ----------
        tools
            a list of available tools. Each tool is defined within a Dict.

        Returns
        ----------
        None
        """

        self.tools = [
            {
                "BUILTIN": "MEM",
                "description": "Useful when we should know the answer or be able to answer from memory, or the question is self-contained",
                "dynamic_params": {},
                "method": "GET",
                "params": [],
                "examples": [
                    ("Who did I say I was?"),
                    ("Where are you?"),
                    ("the natural language query."),
                    ("I think your the tits."),
                    ("the open-ended math question."),
                ],
            },
            {
                "BUILTIN": "SPLIT",
                "description": "DO NOT USE",
                "dynamic_params": {},
                "method": "GET",
                "params": [],
                "examples": [],
            },
        ]
        self.bot_str = ""

        for tool in tools:
            if not "examples" in tool:
                tool["examples"] = []
        self.tools += tools

    def use_tool(self, tool, gpt_suggested_input, question, memory):
        """
        Calls the appropriate API based on the tool that's in use.

        Parameters
        ----------
        tool
            an integer refencing the selected tool
        gpt_suggested_input
            the input params for the selected tool as specified by the gpt response
        question
            the user's input question
        memory
            a list of tuples containing the conversation history

        Returns
        ----------
        String
            Response text after calling API tool and passing result into ChatGPT prompt
        """

        return tool_api_call(self, tool, gpt_suggested_input, question, memory)

    def run(self, question, memory):
        """
        Runs the Agent on the given inputs

        Parameters
        ----------
        question
            the user's input question
        memory
            a list of tuples containing the conversation history

        Returns
        ----------
        Tuple
            a tuple containing the Agent's answer and a list of the conversation history
        """

        self.price = 0

        # question = disambiguate(self, question, memory)

        ans = self.promptf(question, memory, "used to respond to the human")

        if self.verbose > -1:
            print_big("Expected Price = ${:.4f}".format(self.price))

        if ans is None:
            ans = ("Can't answer this from context!", [])

        return (ans[0], memory + [(question, ans[0])], ans[1], ans[2])

    def makeInteraction(self, p, a, Q="HUMAN", A="AI", INTERACTION=INTERACTION):
        """
        Formats the tool description to contain the relevant tags according to the dynamic params

        Parameters
        ----------
        p
            the question being asked
        a
            the gpt response
        Q
            the entity asking the question. Could be HUMAN or AI.
        A
            the entity answering the question. Could be HUMAN or AI.
        INTERACTION
            the type of interaction. Could be HUMAN-AI or AI-AI.

        Returns
        ----------
        String
            a formatted string with tags containing the type of interaction, the question and the gpt response
        """

        return f"<{INTERACTION}><{Q}>{p}</{Q}><{A}>" + (
            f"{a}</{A}></{INTERACTION}>" if a is not None else ""
        )

    def promptf(self, question, memory, utility):
        """
        Formats the gpt prompt to include conversation history and logs responses to the console

        Parameters
        ----------
        question
            the user's input question
        memory
            a list of tuples containing the conversation history
        utility
            a string that's appended to the prompt to indicate the user input that gpt is responding to

        Returns
        ----------
        Tuple
            a tuple containing the gpt response and a list with the conversation history
        """

        if self.verbose > 2:
            print_op("OrigQ:", question)

        # d_question = disambiguate(self, question, memory)

        tools_left = dict((i, i) for i in range(0, len(self.tools)))

        while len(tools_left.keys()) > 0:
            # print_op("MEM:", memory)
            # print_op("QQ:", question)
            tool_to_use = choose_tool(self, memory, question, tools_left)

            if tool_to_use is None:
                return None

            try:
                del tools_left[tool_to_use]
            except:
                pass

            tool = self.tools[tool_to_use]

            print_big(
                "".join(
                    [
                        f'{"✓" if self.tools.index(tool) == tool_to_use else " "} {self.tools.index(tool)}.- {tool["description"]}\n'
                        for tool in self.tools
                    ]
                )
                + f"\n> Question: {question}\n> Raw answer: '{tool_to_use}'\n> Tool ID: {tool_to_use}",
                "LIST OF DATA TOOLS",
            )

            tool_tup = (tool_to_use, tool)

            api_calls = []
            full_calls = []
            tool_inputs = {}
            qnum = "Q" + str(len(memory))

            for param in tool["dynamic_params"].items():
                param_name = f"PARAM{tool_to_use}.{qnum}"

                memp = memory + [
                    (
                        qnum + ": " + question,
                        f"Thinking..\n"
                        # + f"{param_name} DESC: {param[1]}."
                    )
                ]
                # subq = f"Find {param_name}</VAL> to use a tool with <DESC>{tool['description']}</DESC> to answer {qnum}"

                subq = f"{param[1]}"
                # subq = new_subq(self, memory, question, tool['description'], param[1])

                output = self.promptf(
                    subq,
                    memp,
                    f"used as input for a tool described as <DESC>{tool['description']}</DESC> to answer {qnum}.",
                )

                if output is None:
                    return None
                tool_input, new_api_calls, new_full_calls = output

                tool_input = extract(self, param[1], tool_input)

                api_calls += new_api_calls
                full_calls += new_full_calls
                tool_inputs[param[0]] = tool_input

            ret = None
            if not "BUILTIN" in tool:
                ret, actual_url, actual_call = self.use_tool(
                    tool, tool_inputs, question, memory
                )
                api_calls += [actual_url]
                full_calls += [actual_call]
            elif tool["BUILTIN"] == "SPLIT":

                def makeEx(q, subq1, subq2):
                    # + "<MEM>" + "<P/>".join(f"{p}: <AI>{a}</AI>" for q,a in mem) + "</MEM>" \
                    tot = (
                        f"<EX>"
                        + f"<Q>{q}</Q>"
                        + f"<SUB_Q>"
                        + (f"{subq1}</SUB_Q>" if subq1 is not None else "")
                        + (
                            f"<GOAL_Q>{subq2}</GOAL_Q></EX>"
                            if subq2 is not None
                            else ""
                        )
                    )

                    if subq1 == None:
                        print_op("TOT:", tot)
                    return tot

                exs = [
                    makeEx(
                        q="What's open right now?",
                        subq1="What time is it?",
                        subq2=f"What's open at the time you responded in {qnum}?",
                    ),
                    makeEx(
                        q="Which is larger: the distance between NYC and Zurich or that between SF and Milan?",
                        subq1="What's the distance between NYC and Zurich?",
                        subq2=f"What's larger: the answer to {qnum} or the distance between SF and Milan?",
                    ),
                    makeEx(
                        q="How much higher is the 10934ft than mount everest",
                        subq1="How high is mount eversest?",
                        subq2=f"How much higher is 10934ft than the answer to {qnum}?",
                    ),
                    makeEx(
                        q="First answer me a riddle about a cat then turn it into hexidecimal",
                        subq1="Answer me a riddle about a cat",
                        subq2=f"Turn the answer to {qnum} into hexidecimal",
                    ),
                ]

                subqs = (
                    call_gpt(
                        self,
                        "\n".join(exs) + "\n" + makeEx(question, None, None),
                        "</GOAL_Q>",
                        max_tokens=500,
                        quality="best",
                        temperature=0.0,
                    )
                    .strip()
                    .split("</SUB_Q><GOAL_Q>")
                )

                print_op("SUBQS:", subqs)

                ans1, new_api_calls, new_private_args = self.promptf(subqs[0], memory)

                print_op("SUBQS_AGAIN!:", subqs)
                ans2, newer_api_calls, newer_private_args = self.promptf(
                    subqs[1], memory + [(qnum + ": " + subqs[0], ans1)]
                )

                memory = memory + [(subqs[0], ans1), (subqs[1], ans2)]
                api_calls += new_api_calls + newer_api_calls
                full_calls += new_private_args + newer_private_args
                # return ans,  new_api_calls + newer_api_calls, new_private_args + newer_private_args

            prompt = (
                MSG(
                    "system",
                    "The assistant is friendly, but kurt and reserved. It only provides exactly the information requested, never more than that. Doesn't appologize, and doesn't mind doing things again.",
                )
                + flatten([MSG("user", p) + MSG("assistant", a) for p, a in memory])
                + MSG("user", question + " (" + utility + ")")
                + (
                    []
                    if ret is None
                    else MSG("assistant", f"Using an intermediate tool... \n{ret}")
                    + MSG(
                        "user",
                        f"Ignoring that I might be asking you this again, convert that to a plain language response to the previous question:",
                    )
                )
            )

            answer = call_ChatGPT(
                self, prompt, max_tokens=1000, temperature=0.0, gpt4=False
            )

            if self.verbose > 2:
                print_op("Q:", question)
                print_op("I:", tool_inputs)
                print_op("A:", answer)
            return (answer, api_calls, full_calls)


# print_op(google(' {"question": ""}'))
if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]
    tools = []
    a = Agent(openai.api_key, buildGenericTools(GENERIC_TOOLS) + tools, verbose=1)
    mem = []
    last = ""
    while True:
        inp = input(last + "Human: ")
        ret = a.run(inp, mem)
        mem = ret[1]
        last = "AI: " + str(ret[0]) + "\n"
