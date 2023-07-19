import json
import os
from llm_vm.utils.typings_llm import *

def __question_splitter_data() -> QuestionSplitInputModel:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(current_dir, f'../../raw_data/question_split_data.json'))

    json_data: List[QuestionSplitModelJSONData] = json.load(open(file_path, "r"))["data"]

    question_data: List[QuestionSplitModelData] = [
        {
            "mem": [(a[0], a[1]) for a in q["mem"]] if "mem" in q else [],
            "question": q["question"],
            "answer": q["answer"]
        }
        for q in json_data
    ]

    return {
        "openai_model": OpenAIModel.DAVINCI,
        "data": question_data
    }

question_splitter_data = __question_splitter_data();
