from llm_vm.agents.FLAT.agent_helper.tool_utils import create_memory_prompt
from llm_vm.agents.FLAT.agent_helper.labels import *
from typings_llm import *
from llm_vm.agents.FLAT.models.utils.answer_from_memory_model.answer_from_memory_model_data import answer_from_memory_data

def __construct_answer_from_memory_jsonl(data: List[QuestionSplitModelData]) -> List[PromptModelEntry]:
    jsonl_entries: List[PromptModelEntry] = []
    for entry in data:
        prompt, stopper = create_memory_prompt([{
            "mem": entry["mem"] if "mem" in entry else None,
            "question": entry["question"],
            "answer": entry["answer"]
        }])

        jsonl_entries.append({
            "prompt": prompt,
            "completion": ("yes" if entry["answer"] else "no") + (stopper if stopper else "")
        })
    return jsonl_entries


def answer_from_memory_jsonl() -> Tuple[PromptModelEntry, str]:
    return (
        __construct_answer_from_memory_jsonl(answer_from_memory_data["data"]),
        answer_from_memory_data["openai_model"].value
    )
