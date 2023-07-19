from llm_vm.utils.typings_llm import *
import json
import os
from llm_vm.agents.FLAT.agent_helper.tool_utils import get_training_tool_subset
from llm_vm.agents.FLAT.agent_helper.tools import GENERIC_TOOLS
from llm_vm.agents.FLAT.models.utils.tool_picker_model.get_training_tools import get_randomised_training_tools
from random import randrange

def __use_tool(tool_id: int, shuffle_value: int = 0, shuffle_modulo: int = 1e9) -> int:
    return int((tool_id + shuffle_value) % shuffle_modulo)

# Get a random subset of the tools, which WILL INCLUDE the given tool_id.
# -> The tools are shuffled to prevent the model from finding patterns
# -> The ids are changed to prevent the model from remembering tools (i.e. Tool=X is always ID=Y)
def __get_random_tool_subset(tool_id: int, shuffle_value: int = 0, shuffle_modulo: int = 1e9) -> str:
    return get_training_tool_subset(
        get_randomised_training_tools(GENERIC_TOOLS, shuffle_value, shuffle_modulo),
        tool_id
    )

def __get_toolpicker_model(tools_input_model: ToolpickerInputModel) -> ToolpickerInputModel:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(current_dir, f'../../raw_data/toolpicker_data.json'))
    json_data: List[QuestionSplitModelJSONData] = json.load(open(file_path, "r"))["data"]

    tools_input_model["data"] = json_data
    complete_tools: List[ToolpickerInputModelData] = []
    for t in tools_input_model["data"]:
        rand_value = randrange(0, len(tools_input_model["data"]))
        answer_key = t["answer"]
        answer_value = ALL_TOOLS_MAP[answer_key].value if answer_key in ALL_TOOLS_MAP else None
        if (answer_value == None):
            raise Exception("Invalid tool name/value: " + answer_key)

        # tool_id_to_use = __use_tool(answer_value, rand_value) if answer_value > 0 else answer_value
        tool_id_to_use = 0
        complete_tools.append({
            "question": t["question"],
            "thought": t["thought"] if "thought" in t else "",
            "mem": [(a[0], a[1]) for a in t["mem"]] if "mem" in t else [],
            "answer": tool_id_to_use,
            "tools": __get_random_tool_subset(tool_id_to_use, rand_value)
        })

    complete_model: ToolpickerInputModel = {
        "openai_model": tools_input_model["openai_model"],
        "data": complete_tools
    }
    return complete_model;

tool_input_data: ToolpickerInputModel = __get_toolpicker_model({
    "openai_model": OpenAIModel.DAVINCI
})
