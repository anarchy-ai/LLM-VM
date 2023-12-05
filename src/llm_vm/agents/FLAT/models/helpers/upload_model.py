import json
import sys
import openai
import time
import os
from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.models.helpers.check_model_status import check_model_status

__START_LABEL    = " 🎛️  Creating model..."
__PROGRESS_LABEL = " ⏳ Processing"
__FINISHED_LABEL = "\n 🎉 Finished!" + \
                   "\n 🔗"

def upload_model (
    data: PromptModelEntry,
    openai_model: str,
    file_name: str,
    is_test: bool = False
) -> LLMTrainingResult:
    start_of_execution = time.time()

    def progress_label(label: str) -> str:
        ellapsed_s = int(time.time() - start_of_execution)
        return f"{label} '{file_name}'\t ({'%dm %02ds' % (ellapsed_s / 60, ellapsed_s % 60)}) "

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f'./../data/{file_name}.jsonl')

    with open(file_path, "w") as output_file:
        output_file.truncate(0)
        for entry in data:
            json.dump(entry, output_file)
            output_file.write("\n")

    if is_test:
        return {
            "model_name": file_name + "_EXXX",
            "model_files": [],
            "elapsed_time_s": int(time.time() - start_of_execution)
        }

    upload_response = openai.files.create(file=open(file_path, "rb"),
    purpose='fine-tune')

    file_id = upload_response.id

    fine_tune_response = openai.fine_tunes.create(training_file=file_id, model=openai_model)
    job_id = fine_tune_response.id
    print(__START_LABEL, f'''job_id: {job_id}, model: {file_name}''', flush=True, file=sys.stderr)

    finetuned_model = {"status": None}
    while finetuned_model["status"] not in ["succeeded", "failed"]:
        time.sleep(15)
        finetuned_model = check_model_status(job_id, progress_label(__PROGRESS_LABEL))

    print(progress_label(__FINISHED_LABEL), finetuned_model["fine_tuned_model"], flush=True, file=sys.stderr)

    related_files = (
        [file["id"] for file in finetuned_model["result_files"]] + \
        [file["id"] for file in finetuned_model["training_files"]]
    )

    return {
        "model_name": finetuned_model["fine_tuned_model"],
        "model_files": related_files,
        "elapsed_time_s": int(time.time() - start_of_execution)
    }
