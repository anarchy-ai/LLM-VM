import llm_vm.client as l 

openai_key = ""
ret = l.complete("Who are the top 10 greatest soccer players", "", openai_key)
assert type(ret) == str
ret = l.complete("Who are the top 10 greatest soccer players", "", "")
assert type(ret) == Exception
ret = l.complete("What should lions do if they are hungry", "", openai_key, finetune=0)
assert type(ret) == Exception
ret = l.complete("What should lions do if they are hungry", "", openai_key, finetune=True, data_synthesis= 90)
assert type(ret) == Exception
ret = l.complete("", "", openai_key, finetune=True, data_synthesis= 90)
assert type(ret) == Exception
ret = l.complete("What is 2+2+2", "Answer the following math problem with just a number", openai_key, finetune=True)
assert type(ret) == str
assert ret.strip() == "6"
print("All tests passed")