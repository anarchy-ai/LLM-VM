import json
from llm_vm.utils.typings_llm import *
import os
import time

def persist_models(models: LLModels, openai_key: str, is_test: bool = False) -> None:
    if is_test:
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, '../models.json')

    model_json: List[LLModels] = []
    with open(file_name, "r") as output_file:
        loaded_json = json.loads(output_file.read() or "[]")
        model_json: List[LLModels] = loaded_json if loaded_json else []

    with open(file_name, "w") as output_file:
        models["TIMESTAMP"] = int(time.time())
        models["DATE"] = time.asctime()
        models["OPENAI_KEY"] = openai_key
        model_json.append(models)
        json.dump(model_json, output_file, indent=4)
