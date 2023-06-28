import openai
from typings_llm import *

def check_model_status (
    job_id: str,
    label: str
) -> Dict:
    finetuned_model = openai.FineTune.retrieve(id=job_id)
        
    print(
        label, 
        finetuned_model["status"], 
        flush=True
    )
    
    return finetuned_model
