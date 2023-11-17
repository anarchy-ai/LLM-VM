import re
import random
from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.agent_helper.requests.call_llm import call_llm

from llm_vm.agents.FLAT.agent_helper.tools import *
from llm_vm.agents.FLAT.agent_helper.tool_utils import *
from llm_vm.agents.FLAT.agent_helper.utils import *

from llm_vm.agents.FLAT.models.get_decision_model import get_newest_decision_model
from llm_vm.agents.FLAT.models.utils.tool_picker_model.tool_picker_model_data import tool_input_data
from llm_vm.agents.FLAT.models.utils.question_split_model.question_split_model_data import question_splitter_data

# openai.api_key = OPENAI_DEFAULT_KEY


def question_split(question_to_split, use_fine_tuned_model = True):

    question_data = question_splitter_data["data"].copy()
    random.shuffle(question_data)
    NUMBER_OF_EXAMPLES = 6
    question_data = question_data[:NUMBER_OF_EXAMPLES]
    use_fine_tuned_model = False
    prompt, prompt_stop = splitter_prompt(
        question_data +
        [question_to_split]
    )

    if use_fine_tuned_model:
            custom_model, custom_model_key = get_newest_decision_model(DecisionStep.SPLIT)
    else:
        custom_model, custom_model_key = OpenAIModel.DAVINCI_TEXT.value, None

    sub_questions_str, price = call_llm({
        "llm": LLMCallType.OPENAI_COMPLETION,
        "model": (custom_model, custom_model_key),
        "max_tokens": 256,
        "stop": prompt_stop,
        "prompt": prompt,
        "temperature": 0.2,
    })

    sub_questions, main_question = tidy_up_subquestions(
        sub_questions_str.replace(prompt_stop, ""),
        question_to_split["question"]
    )

    return sub_questions, main_question, price


def pick_tool(tools_list, question, conversation_history, use_fine_tuned_model = True, debug_prompt = False):
    tool_data = tool_input_data["data"].copy()
    random.shuffle(tool_data)
    NUMBER_OF_EXAMPLES = 6
    tool_data = tool_data[:NUMBER_OF_EXAMPLES]
    use_fine_tuned_model = False


    prompt, stop = toolpicker_prompt(
        tool_data + [{"question": question, "mem": conversation_history}],
        tools_list,
    )

    if debug_prompt:
        print_big(prompt, "toolpicker prompt")

    if use_fine_tuned_model:
        custom_model, custom_model_key = get_newest_decision_model(
            DecisionStep.TOOL_PICKER)
    else:
        custom_model, custom_model_key = OpenAIModel.DAVINCI_TEXT.value, None

    suggested_tool_str, price = call_llm({
        "llm": LLMCallType.OPENAI_COMPLETION,
        "model": (custom_model, custom_model_key),
        "max_tokens": 10,
        "stop": stop,
        "prompt": prompt,
        "temperature": 0,
    })

    number_match_regex = "-?[0-9]+"

    try:
        valid_tool_ids = [t["id"] for t in tools_list] + \
            [DefaultTools.ANSWER_FROM_MEMORY.value, DefaultTools.I_DONT_KNOW.value]
        parsed_suggested_tool_str = re.findall(
            number_match_regex, suggested_tool_str)
        if len(parsed_suggested_tool_str) == 0:
            raise Exception("No matching ID found in response")

        suggested_tool_id = int(parsed_suggested_tool_str[0])
        if suggested_tool_id not in valid_tool_ids:
            raise Exception(
                f"Invalid tool id: {suggested_tool_id} // {','.join([str(v) for v in valid_tool_ids])}")

        tools_list.sort(key=lambda t: t["id"])
        print_big(
            "".join([f'{"✓" if tool["id"] == suggested_tool_id else " "} {tool["id"]}.- {tool["description"]}\n' for tool in tools_list]) +
            f"\n> Question: {question}\n> Raw answer: '{suggested_tool_str}'\n> Tool ID: {suggested_tool_id}",
            "LIST OF DATA TOOLS"
        )

        return suggested_tool_id, price
    except Exception as e:
        print_big(f'Exception parsing tool_id: "{suggested_tool_str}"')
        print(e, flush=True, file=sys.stderr)
        return DefaultTools.I_DONT_KNOW.value, price


def check_can_answer_from_memory(question: str, memory: TupleList = [], facts: TupleList = []) -> Tuple[bool, float]:

    memory_prompt, memory_stop = create_memory_prompt(
        # answer_from_memory_data["data"] +
        [{"question": question, "mem": memory, "facts": facts}]
    )

    ans, price = call_llm({
        "llm": LLMCallType.OPENAI_COMPLETION,
        "model": get_newest_decision_model(DecisionStep.FROM_MEMORY),
        "max_tokens": 10,
        "stop": memory_stop,
        "prompt": memory_prompt,
        "temperature": 0.1,
    })

    if "yes" in ans.lower():
        return True, price
    else:
        return False, price
