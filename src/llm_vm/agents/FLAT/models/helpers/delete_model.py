import openai

def delete_model (model_name: str) -> None:
    ## DELETE MODELS
    try:
        # model = openai.Model.retrieve()
        openai.Model.delete(model_name)
    except Exception as e:
        print("Could not delete model because: " + str(e))
        
        
    try:
        # Delete all files:
        files = openai.File.list()["data"]
        [openai.File.delete(file["id"]) for file in files];
    except Exception as e:
        print("Could not delete file because: " + str(e))