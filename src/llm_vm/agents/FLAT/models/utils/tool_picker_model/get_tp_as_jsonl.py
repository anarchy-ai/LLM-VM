from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.agent_helper.labels import *
from llm_vm.agents.FLAT.agent_helper.tool_utils import toolpicker_prompt
from llm_vm.agents.FLAT.models.utils.tool_picker_model.tool_picker_model_data import tool_input_data

def __construct_tool_picker_jsonl(data: List[ToolpickerInputModelData]) -> List[PromptModelEntry]:
    jsonl_entries: List[PromptModelEntry] = []
    for entry in data:
        prompt, stopper = toolpicker_prompt(
            [{
                "mem": entry["mem"] if "mem" in entry else None,
                "question": entry["question"],
                "thought": entry["thought"] if "thought" in entry else None
            }],
            entry["tools"]
        )
        jsonl_entries.append({
            "prompt": prompt,
            "completion": str(entry["answer"]) + stopper
        })
    return jsonl_entries


def tool_picker_jsonl() -> Tuple[PromptModelEntry, str]:
    return (
        __construct_tool_picker_jsonl(tool_input_data["data"]),
        tool_input_data["openai_model"].value
    )
