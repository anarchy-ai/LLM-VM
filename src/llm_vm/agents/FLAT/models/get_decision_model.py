import sys
from llm_vm.utils.typings_llm import *
import os
import json

def get_newest_decision_model(model: DecisionStep, default_model = OpenAIModel.DAVINCI_TEXT) -> Tuple[str, Union[str, None]]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(current_dir, 'models.json')

    model_name = model.value
    default_model_name = default_model.value
    print(file_name, file=sys.stderr)
    file = open(file_name)

    data: List[LLModels] = json.load(file);

    for llm_model in reversed(data):
        if model.value in llm_model and llm_model[model_name]:
            print(f"### {model_name} model will be used (version {llm_model['DATE']})", flush=True, file=sys.stderr)
            return llm_model[model_name]['model_name'], llm_model['OPENAI_KEY']

    print(f"Model {model_name} not found: will use {default_model_name}", flush=True, file=sys.stderr)
    return default_model_name, None
