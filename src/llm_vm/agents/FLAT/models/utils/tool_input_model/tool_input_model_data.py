from llm_vm.utils.typings_llm import *
import os
import json
from llm_vm.agents.FLAT.agent_helper.utils import verbose_answer

def __get_tool_input_model(model: OpenAIModel) -> ToolInputModel:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(current_dir, f'../../raw_data/tool_input_data.json'))

    json_data: List[ToolInputModelJSONData] = json.load(open(file_path, "r"))["data"]

    def __get_mem_tuple(json_mem: List[List[Union[str, dict]]]) -> TupleList:
        mem: TupleList = []
        for m in json_mem:
            if len(m) == 3:
                # when the "mem" has also the previous JSON response
                mem.append((m[0], verbose_answer(m[2], m[1])))
            elif len(m) == 2:
                # when the "mem" has only question/answer
                mem.append((m[0], m[1]))
        return mem

    return {
        "openai_model": model,
        "data": [{
            "mem": __get_mem_tuple(t["mem"]) if "mem" in t else [],
            "question": t["question"],
            "answer": t["answer"],
            "description": t["description"],
            "params": t["params"]
        } for t in json_data]
    }


tool_input_data: ToolInputModel = __get_tool_input_model(OpenAIModel.DAVINCI)
