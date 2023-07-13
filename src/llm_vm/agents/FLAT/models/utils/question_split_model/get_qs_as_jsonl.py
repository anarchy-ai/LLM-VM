from llm_vm.agents.FLAT.agent_helper.tool_utils import splitter_prompt
from llm_vm.agents.FLAT.agent_helper.labels import *
from llm_vm.typings_llm import *
from llm_vm.agents.FLAT.models.utils.question_split_model.question_split_model_data import question_splitter_data

def __construct_question_split_jsonl(data: List[QuestionSplitModelData]) -> List[PromptModelEntry]:
    jsonl_entries: List[PromptModelEntry] = []
    for entry in data:
        prompt, stopper = splitter_prompt([{
            "mem": entry["mem"] if "mem" in entry else None,
            "tools": entry["tools"] if "tools" in entry else None,
            "question": entry["question"]
        }])
        jsonl_entries.append({
            "prompt": prompt,
            "completion": SUBQ_SPLITTER.join(entry["answer"]) + stopper
        })
    return jsonl_entries


def question_splitter_jsonl() -> Tuple[PromptModelEntry, str]:
    return (
        __construct_question_split_jsonl(question_splitter_data["data"]),
        question_splitter_data["openai_model"].value
    )
