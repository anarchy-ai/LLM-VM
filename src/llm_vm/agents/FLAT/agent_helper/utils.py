import traceback
import os
import sys
import re
import openai

# Get the current file's directory to grab the python files with common functionality in the utils/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
utils_dir = os.path.join(grandparent_dir, 'utils/')
sys.path.append(utils_dir)

from llm_vm.utils.labels import *
import random
from Levenshtein import distance as lev
from llm_vm.utils.typings_llm import *
from bs4 import BeautifulSoup


random_fixed_seed = random.Random(4)

def print_op(*kargs, **kwargs):
    print(*kargs, **kwargs, flush=True, file=sys.stderr)

def verbose_answer(data, answer):
    return f'''<{L_ANSWER_DATA}>{str(data)}</{L_ANSWER_DATA}><{L_ANSWER_SUMMARY}>{answer}</{L_ANSWER_SUMMARY}>'''

def remove_tags_from_html_string(html_string):
    # if not isinstance(html_string, str):
    #     return ""
    # return re.compile(r'<[^>]+>').sub("", html_string), False


    soup = BeautifulSoup(html_string, 'html.parser')
    has_friendly_tags: bool = False

    # Find all tags that are not in the list of tags to preserve
    for tag in soup.find_all(True):
        if tag.name in ['ul', 'ol', 'li', 'a', 'pre', 'code', 'kbd', 'i']:
            has_friendly_tags = True
        elif tag.name in ['style', 'script']:
            tag.extract() # remove tag and content
        else:
            tag.unwrap() # remove tag, but preserve content

    # Get the stripped HTML string
    return str(soup).strip("\n "), has_friendly_tags

def print_big(data, label = ""):
    def do_format(x) -> str:
        formatted_title = "======#====== {:20} ======#======\n"
        if len(x) >= 20:
            return formatted_title.format(x)
        else:
            return formatted_title.format((int((20 - len(x)) / 2) * " ") + x)
    try:
        if len(label):
            print(do_format(str(label).upper()), data, flush=True, file=sys.stderr)
        else:
            print(do_format(str(data)), flush=True, file=sys.stderr)

    except:
        print(label, flush=True, file=sys.stderr)


# Tolerance: If string A can be converted into B in more than {tolerance} steps, they are considered different.
def remove_similars(similars_list, tolerance = 3):
    unique_list = []
    sanitized_list = list(map(lambda item: item.strip(" ").lower(), similars_list))
    similars_len = len(similars_list)

    for i in range(similars_len):
        has_duplicate: bool = False
        for j in range(i+1, similars_len):
            if lev(sanitized_list[i], sanitized_list[j], score_cutoff=tolerance) < tolerance:
                has_duplicate = True
                break
        if has_duplicate == False:
            unique_list.append(similars_list[i])
    return unique_list

def make_interaction_request(human_question, ai_response, data, Q = L_QUESTION, A = L_ANSWER, D = L_ANSWER_DATA, INTERACTION = L_INTERACTION):

    interaction_text = f"<{INTERACTION}>\n<{Q}>{human_question}</{Q}>\n"

    if data and len(data) and isinstance(data, str):
        interaction_text += f'''<{D}>{data}</{D}>'''

    interaction_text += f'''<{A}>{ai_response if ai_response else ''}'''

    stop = f"</{A}>\n</{INTERACTION}>"

    return interaction_text, stop

def make_interaction(human_question, ai_response, data = '',
                     Q = L_QUESTION,
                     A = L_ANSWER,
                     D = L_ANSWER_DATA,
                     INTERACTION = L_INTERACTION):
    text, stop = make_interaction_request(human_question, ai_response, data, Q, A, D, INTERACTION)
    return text + stop

def get_tool_by_id(tools, tool_id):
    for tool in tools:
        if tool['id'] == tool_id:
            return tool
    raise Exception(f"No tool corresponds to id='{tool_id}'")

def tidy_up_subquestions(subquestions_str, main_question):
    sub_questions_raw = subquestions_str.replace("\n", "").split(SUBQ_SPLITTER)

    # First, clear the subquestions of extra symbols that might be confusing
    sub_questions_raw = list(filter(lambda q: len(q) > 1, [q.strip() for q in sub_questions_raw]))
    # then, remove similar subquestions
    unique_subquestions = remove_similars(sub_questions_raw)

    # if we only have one sub question, we don't need to "summarise" at the end => main question == subquestion.
    if len(unique_subquestions) == 1:
        return [main_question], None
    else:
        # to make sure, remove similar subquestions again because we add the main question
        # sub_questions = remove_similars(sub_questions + [question])
        return unique_subquestions, main_question
