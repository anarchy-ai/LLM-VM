import openai
import re


def tool_picker(tools_list, question, starting_tool_num):

    tools=""
    prompt= '''
{tools}
Which tool (number only), if any, would you use to answer the following question:

{question}
    '''
    count=0
    tools_list =  tools_list[starting_tool_num:]
    for i in tools_list:
        tools+="tool "+str(count)+": "+str(i["description"])+"\n"
        count+=1
    tools+="tool "+str(count)+": "+str("Use this tool when the question can be answered without using any tool or if the question is a greeting or a casual conversation. Also if we need to convert languages other than english, use this tool.")+"\n"
    prompt=prompt.format(**{"tools":tools,"question":question})
    prompt += "\n\nYOU JUST ANSWER WITH A NUMBER."
    ans = openai.completions.create(model="text-davinci-003",
    max_tokens=256,
    stop=None,
    prompt=prompt,
    temperature=0.4)
    
    try:
        return (calcCost(prompt),str(int(re.sub("[^0-9]", "",ans['choices'][0]['text'].strip()))+starting_tool_num))
    except:
        return  (calcCost(prompt),1+starting_tool_num)

def calcCost(p):
            return (len(p) / 2700.0) * 0.02

def question_split(question,tools_list,memory):
    tools=""
    starting_tool_num=2
    count=starting_tool_num
    for i in range(len(tools_list)-starting_tool_num):
        tools+="tool "+str(i+starting_tool_num)+": "+str(tools_list[starting_tool_num+i]["description"])+"\n"
        count+=1
    tools+="tool "+str(count)+": "+str("Use this tool when the question can be answered without using any tool or if the question is a greeting or a casual conversation.")+"\n"

    prompt='''
Tools we have access to =
tool 1: The tool returns the results of free-form queries similar to those used for wolfram alpha. This is useful for complicated math or live data retrieval.  Can be used to get the current date.
tool 2: Find the driving distance and time to travel between two cities.
tool 3: Find the weather at a location and returns it in celcius.

Q="Is the distance between London and Paris larger than the distance between Boston and LA?"
Look at the tools we have access to. Split Q into subquestions to answer Q that can each be solved with one use of one tool. Make as few subquestions as possible. Split each subquestion with a comma and have no extra information other than the subquestions.
What is the distance from London to Paris?, What is the distance from Boston to LA?

Tools we have access to =
tool 1: The tool returns the results of free-form queries similar to those used for wolfram alpha. This is useful for complicated math or live data retrieval.  Can be used to get the current date.
tool 2: Find the driving distance and time to travel between two cities.
tool 3: Find the weather at a location and returns it in celcius.

Q="Who was the maternal grandfather of george washington?"
Look at the tools we have access to. Split Q into subquestions to answer Q that can each be solved with one use of one tool. Make as few subquestions as possible. Split each subquestion with a comma and have no extra information other than the subquestions.
Who was george washington's mother?,Who was her father?

Tools we have access to =
tool 1: The tool returns the results of free-form queries similar to those used for wolfram alpha. This is useful for complicated math or live data retrieval.  Can be used to get the current date.
tool 2: Find the driving distance and time to travel between two cities.
tool 3: Find the weather at a location and returns it in celcius.

Q:"What is the currency in India"
Look at the tools we have access to. Split Q into subquestions to answer Q that can each be solved with one use of one tool. Make as few subquestions as possible. Split each subquestion with a comma and have no extra information other than the subquestions.


Tools we have access to =
tool 1: The tool returns the results of free-form queries similar to those used for wolfram alpha. This is useful for complicated math or live data retrieval.  Can be used to get the current date.
tool 2: Find the driving distance and time to travel between two cities.
tool 3: Find the weather at a location and returns it in celcius.

Q:"If I am in Miami how far am I from Atlanta?"
Look at the tools we have access to. Split Q into subquestions to answer Q that can each be solved with one use of one tool. Make as few subquestions as possible. Split each subquestion with a comma and have no extra information other than the subquestions.


Tools we have access to =
{tools}
Q= "{question}"
Look at the tools we have access to. Split Q into subquestions to answer Q that can each be solved with one use of one tool. Make as few subquestions as possible. Split each subquestion with a comma and have no extra information other than the subquestions.
    '''
    prompt=prompt.format(**{"question":question,"tools":tools,"memory":memory})
    ans = openai.completions.create(model="text-davinci-003",
    max_tokens=256,
    stop=None,
    prompt=prompt,
    temperature=0.2)
    ans=ans['choices'][0]['text'].replace("\n","").split(",")
    return (calcCost(prompt),ans)


def memory_check(memory,question):

    prompt='''
Q: "What's up?"

Is the answer to Q found in the memory or in your knowledge base already? Answer with a yes or no. yes

Q: "What color is the sky"

Is the answer to Q found in the memory or in your knowledge base already? Answer with a yes or no. yes

Q: "What is the temperature in Portland?"

Is the answer to Q found in the memory or in your knowledge base already? Answer with a yes or no. no

Memory:
{memory}

Q: "{question}"

Is the answer to Q found in the memory or in your knowledge base already? Answer with a yes or no.
    '''
    prompt=prompt.format(**{"memory":memory,"question":question})
    ans = openai.completions.create(model="text-davinci-003",
    max_tokens=256,
    stop=None,
    prompt=prompt,
    temperature=0.4)

    ans=ans['choices'][0]['text']

    if "yes" in ans.lower():
        return (calcCost(prompt),True)
    else:
        return (calcCost(prompt),False)

def replace_variables_for_values(my_dict: dict, dynamic_keys, ignore_key: str = "_______"):
    replaced_dict = {}
    for key, value in my_dict.items():
        if (key == ignore_key):
            continue;
        formatted_key = key.format(**dynamic_keys)
        if (isinstance(value, dict)):
            formatted_value = replace_variables_for_values(value, dynamic_keys)
        elif (isinstance(value, list)):
            formatted_value = []
            for item in value:
                formatted_value += replace_variables_for_values(item, dynamic_keys)
        else:
            try:
                formatted_value = value.format(**dynamic_keys)
            except:
                formatted_value = value
        replaced_dict[formatted_key] = formatted_value
    return replaced_dict
#print(question_split(""))
