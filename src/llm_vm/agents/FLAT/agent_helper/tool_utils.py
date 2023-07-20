import json
import os
import sys

# # Get the current file's directory to grab the python files with common functionality in the utils/ folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# utils_dir = os.path.join(grandparent_dir, 'utils/')
# sys.path.append(utils_dir)

from llm_vm.utils.labels import *
from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.agent_helper.utils import make_interaction
from datetime import datetime
from llm_vm.agents.FLAT.agent_helper.tools import GENERIC_TOOLS
from random import sample, shuffle


def __create_tool_tag(tool: SingleTool) -> str:
    return f'<{L_TOOL} id="{tool["id"]}" description="{tool["description"]}"/>'

__DEFAULT_TOOLS = "\n".join(
    [__create_tool_tag(tool)
     for tool in GENERIC_TOOLS]
)

def __tools_we_have_access_to(tools: str = __DEFAULT_TOOLS) -> str:
    return f"""<{L_TOOL_LIST}>{tools}</{L_TOOL_LIST}>"""


def __split_into_subquestions_instructions(extra: str = "") -> str:
    return f"""<{L_SPLIT_INSTRUCTIONS}>Look at the tools we have access to. If needed, split <{L_QUESTION}/> into subquestions, so each <{L_QUESTION}/> can be answered with one use of one tool. Make as few subquestions as possible, comma-separated, avoid duplicates, and return nothing else.{extra}</{L_SPLIT_INSTRUCTIONS}>"""

def make_tool_desc(tool: SingleTool):
    params = "{" + ", ".join(['"'+l + '": '+v for l, v in tool['dynamic_params'].items()]
                               )+"}" if tool['dynamic_params'] != "" else "{}"
    return f'''<{L_TOOL} id="{tool['id']}">
  <{L_DESCRIPTION}>{tool['description']}</{L_DESCRIPTION}>
  <{L_PARAMS}>{params}</{L_PARAMS}>
</{L_TOOL}>
'''

def generate_convo_history(memory: TupleList = [], facts: TupleList = []) -> str:
    if len(memory) + len(facts) == 0:
        return ""

    return (
        "\n".join([make_interaction(p, a, '', L_QUESTION, L_ANSWER, L_ANSWER_DATA, INTERACTION = L_INT_HUMAN) for p, a in memory]) + \
        "\n".join([make_interaction(p, a, '', L_QUESTION, L_ANSWER, L_ANSWER_DATA, INTERACTION = L_INT_AI) for p, a in facts])
    )

def prompt_for_instructions(instructions: str) -> str:
    if instructions and len(instructions):
        return f'''[{L_BOT_INSTRUCTIONS}]: To summarize the JSON, remember:\n {instructions}\n\n\n'''
    else:
        return ""
def prompt_for_answer(question: str) -> str:
    return f'''[{L_QUESTION}]: {question}\n[{L_ANSWER}]: ''';

## ONLY for training purposes!
def get_training_tool_subset (
    list_of_tools: ToolList,
    tool_id_always_returned: Union[int, None],
    max_num_elements: int = 6
) -> ToolList:
    # ONLY when we don't know, we return any elements.
    if tool_id_always_returned == None \
       or tool_id_always_returned == DefaultTools.I_DONT_KNOW.value \
       or tool_id_always_returned == DefaultTools.ANSWER_FROM_MEMORY.value:
        return sample(list_of_tools, max_num_elements - 1)

    specific_tool = next(tool for tool in list_of_tools if tool["id"] == tool_id_always_returned)
    subset_of_tools = sample(list_of_tools, max_num_elements - 1)
    # make sure it doesn't have our specific_tool inside
    subset_of_tools = [tool for tool in subset_of_tools if tool["id"] != tool_id_always_returned] + [specific_tool]

    # VERY IMPORTANT - otherwise the model thinks that the last answer is always the correct one.
    shuffle(subset_of_tools)

    return subset_of_tools


def splitter_prompt(elements: List[QuestionSplitModelData]) -> Tuple[str, str]:

    stopper: str = f'''</{L_ANSWER}>
</{L_SPLIT}>'''

    last_stop = None;

    response: str = f'''\
{__split_into_subquestions_instructions()}'''

    for element in elements:
        response += f'''\
<{L_SPLIT}>
{
    __tools_we_have_access_to("".join([__create_tool_tag(tool) for tool in element["tools"]])) if ("tools" in element and element["tools"] and isinstance(element["tools"], list)) else ""
}{
    f"""<{L_CONVERSATION}>{"".join([make_interaction(q,a) for (q,a) in element["mem"]])}</{L_CONVERSATION}>""" if ("mem" in element and isinstance(element["mem"], list)) else ""
}
<{L_QUESTION}>{element["question"]}</{L_QUESTION}>
<{L_ANSWER}>'''

        if "answer" in element and element["answer"]:
            response += f'''{SUBQ_SPLITTER.join(element["answer"])}{stopper}\n'''
            last_stop = None
        else:
            last_stop = stopper

    return response, last_stop


def toolpicker_prompt(elements: List[ToolpickerInputModelData], tools: ToolList) -> str:

    shuffled_tools = tools.copy()
    shuffle(shuffled_tools)
    tools_html = "\n".join([__create_tool_tag(_tool) for _tool in shuffled_tools])
    last_stop = None

    prompt: str = f'''<{L_BOT_INSTRUCTIONS}><{L_DESCRIPTION}>
We have a list of tools in <{L_TOOL_LIST}/>, and a question <{L_QUESTION}/> that we want to answer.
Which of the <{L_TOOL}/> in <{L_TOOL_LIST}/> can be used to best answer <{L_QUESTION}/>?
Answer only with the 'id' of the most relevant <{L_TOOL}/> (i.e.: {", ".join([str(t['id']) for t in shuffled_tools])}).
Answer {DefaultTools.ANSWER_FROM_MEMORY.value} if the answer is in <{L_CONVERSATION}/>.
Answer {DefaultTools.I_DONT_KNOW.value} if no tool can be used to answer <{L_QUESTION}/>.
</{L_DESCRIPTION}>

<{L_TOOL_LIST}>
{tools_html}
</{L_TOOL_LIST}>
</{L_BOT_INSTRUCTIONS}>
'''

    for element in elements:

        prompt += f'''<{L_INTERACTION}>'''

        prompt += f'''<{L_CONVERSATION}>
            {"".join([make_interaction(mem_q, mem_a) for mem_q, mem_a in (element["mem"])
        ])}</{L_CONVERSATION}>''' if "mem" in element and isinstance(element["mem"], list) and len(element["mem"]) else ""

        prompt += f'''{
      f'<{L_THOUGHT}>{element["thought"]}</{L_THOUGHT}>' if "thought" in element and element["thought"] else ""
}
<{L_QUESTION}>{element["question"]}</{L_QUESTION}>
<{L_ANSWER}>'''

        stopper = f'''</{L_ANSWER}>\n</{L_INTERACTION}>'''

        if "answer" in element:
            prompt += f'''{element["answer"]}{stopper}\n'''
            last_stop = None
        else:
            last_stop = stopper

    return prompt, last_stop


def create_memory_prompt(elements: List[AnswerInMemoryModelData]) -> Tuple[str, str]:
    do_you_know_the_answer = f"<{L_BOT_INSTRUCTIONS}>Is the answer to <{L_QUESTION}/> found in the memory or in your knowledge base already? Answer with a yes or no.</{L_BOT_INSTRUCTIONS}>"

    stop = f"</{L_ANSWER}></{L_INTERACTION}>"

    last_stop = None

    prompt: str = f'''{do_you_know_the_answer}\n'''

    for element in elements:
        conversation_history = generate_convo_history(
            element["mem"] if "mem" in element and element["mem"] else [],
            element["facts"] if "facts" in element and element["facts"] else []
        )
        prompt += f'''<{L_INTERACTION}>\n'''
        if conversation_history:
            prompt += f'''\
<{L_CONVERSATION}>
    {conversation_history}
</{L_CONVERSATION}>\n'''
        prompt += f'''<{L_QUESTION}>{element["question"]}</{L_QUESTION}>\n<{L_ANSWER}>'''

        if "answer" in element and isinstance(element["answer"], bool):
            prompt += f'''{"yes" if element["answer"] else "no"}{stop}'''
            last_stop = None
        else:
            last_stop = stop
            break

    return prompt, last_stop




def make_tool_input_case(
    q_facts: TupleList,
    q_question: str,
    answer_json_input: Union[dict, None],
    tool_descr: str,
    tool_params: dict,
    answer_label: str = "JSON",
    wrapper_tag: str = L_EXAMPLE,
) -> Tuple[str, Union[str, None]]:
    system_instructions: str = f'''
We have the following {answer_label}: <{answer_label}>{json.dumps(tool_params)}</{answer_label}>. Each key describes what its value is.\n
The {answer_label} is used where: <{L_DESCRIPTION}>{tool_descr}</{L_DESCRIPTION}>\n
We need a {answer_label} object with the same keys as <{answer_label}/> to answer <{L_QUESTION}/>{f""" (we can use the <{L_CONVERSATION}/> for contextual info as well)""" if len(q_facts) else ""}. Must be a valid {answer_label}!'''

    stopper: str = f"</{L_ANSWER}>\n</{wrapper_tag}>"

    mem = generate_convo_history(facts=q_facts)
    return f"\n<{wrapper_tag}>\n" \
        + f"  <{L_BOT_INSTRUCTIONS}>{system_instructions}</{L_BOT_INSTRUCTIONS}>" \
        + (f"  <{L_CONVERSATION}>{mem}\n</{L_CONVERSATION}>\n" if mem else "") \
        + f"  <{L_QUESTION}>{q_question}</{L_QUESTION}>\n" \
        + f"  <{L_ANSWER} type=\"{answer_label}\">"\
        + (( f"{json.dumps(answer_json_input)}{stopper}") if answer_json_input else ""), None if answer_json_input else stopper
