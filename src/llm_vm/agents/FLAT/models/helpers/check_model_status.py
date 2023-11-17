import sys
import openai
from llm_vm.utils.typings_llm import *

def check_model_status (
    job_id: str,
    label: str
) -> Dict:
    finetuned_model = openai.fine_tunes.retrieve(id=job_id)

    print(
        label,
        finetuned_model["status"],
        flush=True, file=sys.stderr
    )

    return finetuned_model
