import json
from llm_vm.agents.FLAT.agent_helper.tool_utils import create_memory_prompt, make_tool_input_case
from llm_vm.agents.FLAT.agent_helper.labels import *
from typings_llm import *
from random import shuffle
from llm_vm.agents.FLAT.models.utils.tool_input_model.tool_input_model_data import tool_input_data

def __get_tool_input_jsonl(data: List[ToolInputModelData]) -> List[PromptModelEntry]:
    jsonl_entries: List[PromptModelEntry] = []

    for entry in data:
        prompt, stopper = make_tool_input_case(
            entry["mem"],
            entry["question"],
            None,
            entry["description"],
            entry["params"],
        )

        jsonl_entries.append({
            "prompt": prompt,
            "completion": json.dumps(entry["answer"]) + stopper
        })
    shuffled_examples = jsonl_entries.copy()
    shuffle(shuffled_examples)
    return shuffled_examples


def tool_input_jsonl() -> Tuple[PromptModelEntry, str]:
    return (
        __get_tool_input_jsonl(tool_input_data["data"]),
        tool_input_data["openai_model"].value
    )
