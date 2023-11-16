import sys
import openai

def delete_model (model_name: str) -> None:
    ## DELETE MODELS
    try:
        # model = openai.Model.retrieve()
        openai.models.delete(model_name)
    except Exception as e:
        print("Could not delete model because: " + str(e), file=sys.stderr)


    try:
        # Delete all files:
        files = openai.files.list()["data"]
        [openai.files.delete(file["id"]) for file in files];
    except Exception as e:
        print("Could not delete file because: " + str(e), file=sys.stderr)
