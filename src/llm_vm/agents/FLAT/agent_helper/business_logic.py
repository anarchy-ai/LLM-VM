import json
from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.agent_helper.utils import *
from llm_vm.agents.FLAT.agent_helper.bothandler import question_split, pick_tool, check_can_answer_from_memory, generate_convo_history, prompt_for_answer, get_newest_decision_model
from llm_vm.agents.FLAT.agent_helper.use_tool import use_tool
from llm_vm.agents.FLAT.agent_helper.tool_utils import make_tool_input_case

# Get the current file's directory to grab the python files with common functionality in the utils/ folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# utils_dir = os.path.join(grandparent_dir, 'utils/')
# sys.path.append(utils_dir)

from labels import *
from llm_vm.agents.FLAT.agent_helper.requests.call_llm import call_llm

__MAX_RETRIES_GUESS_INPUT = 3


def __get_tool_input(
    tool: SingleTool,
    mem: TupleList,
    question: str,
    verbose: int,
    max_tokens: int = 20,
    temperature: float = 0.0
) -> Tuple[dict, float]:

    gpt_prompt, gpt_prompt_stop = make_tool_input_case(
        mem,
        question,
        None,
        tool_descr=tool["description"],
        tool_params=tool["dynamic_params"],
        wrapper_tag=L_INTERACTION
    )

    if verbose > 3:
        print_big(gpt_prompt, f"GPT PROMPT [tool={tool['id']}]")

    gpt_suggested_input, price = call_llm({
        "llm": LLMCallType.OPENAI_COMPLETION,
        "model": get_newest_decision_model(DecisionStep.INPUT),
        "max_tokens": max_tokens,
        "prompt": gpt_prompt,
        "stop": gpt_prompt_stop,
        "temperature": temperature,
    })

    try:
        if gpt_suggested_input[0] != "{":
            gpt_suggested_input = "{" + gpt_suggested_input
        if gpt_suggested_input[-1] != "}":
            gpt_suggested_input += "}"

        parsed_gpt_suggested_input: dict = json.loads(gpt_suggested_input)
        if verbose > 1:
            _has_keys = len(parsed_gpt_suggested_input.keys()) > 0
            print_big(
                parsed_gpt_suggested_input if _has_keys else "(( no input ))", "GPT SUGGESTED INPUT")
        return parsed_gpt_suggested_input, price
    except:
        print_big(gpt_suggested_input, "INVALID GPT INPUT")
        # {"$ PARSING FAILED $": str(e), "$ REASON $": "malformed JSON"}
        return {}, price


def promptf(
    question: str,
    factual_memory: TupleList,
    tools: ToolList,
    bot_instructions: str,
    verbose: int,
    split_allowed: bool = True,
    force_answer_from_memory: bool = False
) -> Tuple[str, TupleList, List[str], DebugCallList, float]:
    price_accumulator = 0

    if split_allowed:
        conversation_memory: TupleList = []
        sub_questions, main_question, price = question_split({
            "mem": factual_memory,
            "question": question,
            # "tools": tools
        })
        price_accumulator += price

        if verbose > 0:
            if len(sub_questions) == 1:
                print_big(f"NO SPLIT REQUIRED")
            else:
                print_big(f'''Question:\n- {main_question}\n\nSubquestions:\n''' + (
                    "- " + "\n- ".join(sub_questions)), f"SPLIT")

        calls: List[str] = []
        debug_return: DebugCallList = []

        for i in range(len(sub_questions)):
            answer, new_facts, new_calls, new_debug_return, price = promptf(
                sub_questions[i],
                factual_memory,
                tools,
                bot_instructions,
                verbose,
                split_allowed=False
            )
            calls = calls + new_calls
            factual_memory = factual_memory + new_facts
            debug_return = debug_return + new_debug_return
            price_accumulator += price
            conversation_memory.append((sub_questions[i], answer))

        if main_question:
            print_big(main_question, "TRYING TO ANSWER MAIN QUESTION")
            final_answer, _, _, _, price = promptf(
                main_question,
                # last round, we give the non-verbose convo memory.
                conversation_memory,
                [],
                bot_instructions,
                verbose,
                split_allowed=False,
                force_answer_from_memory=True
            )
            price_accumulator += price
            conversation_memory.append((main_question, final_answer))

            answer = final_answer
            print_big(answer, "FINAL ANSWER")
            print_big("=========#####=========")

        return answer, [], calls, debug_return, price_accumulator

    if len(tools):
        best_tool_id, price_picker = pick_tool(tools, question, factual_memory)
        price_accumulator += price_picker
        cannot_use_tools_to_answer = best_tool_id in [
            DefaultTools.ANSWER_FROM_MEMORY.value, DefaultTools.I_DONT_KNOW.value]
    else:
        best_tool_id = DefaultTools.ANSWER_FROM_MEMORY.value if force_answer_from_memory else DefaultTools.I_DONT_KNOW.value
        cannot_use_tools_to_answer = True

    # Only in the cases where we DON'T have a tool (answer_from_tool == False) we check if we can answer from memory.
    can_answer_from_memory = force_answer_from_memory
    if cannot_use_tools_to_answer and not force_answer_from_memory:
        can_answer_from_memory, price_check = check_can_answer_from_memory(
            question, memory=factual_memory)
        price_accumulator += price_check

    print_big(
        f"MEM ({can_answer_from_memory}) / TOOL ({not cannot_use_tools_to_answer})")

    if cannot_use_tools_to_answer and can_answer_from_memory:

        cur_prompt = (
            bot_instructions +
            generate_convo_history(memory=factual_memory) +
            f"\n\n" +
            f"\n[SYSTEM]: Use <{L_ANSWER_DATA}/> to answer better, but follow the format of <{L_ANSWER_SUMMARY}/>\n" +
            prompt_for_answer(question)
        )

        answer, price = call_llm({
            "llm": LLMCallType.OPENAI_COMPLETION,
            "model": OpenAIModel.DAVINCI_TEXT.value if "quality" == "best" else OpenAIModel.CURIE_TEXT.value,
            "max_tokens": 200,
            "prompt": cur_prompt,
            "stop": None,
            "temperature": 0.1,
        })

        price_accumulator += price
        if verbose > 0:
            print_op("Question:  ", question)
            print_op("AI Answer: ", answer)

        return answer, [(question, answer)], [], [], price_accumulator

    conversation_history = generate_convo_history(facts=factual_memory)
    # No information in the Conversation memory AND no tool --> Using AI General Knowledge Base.
    if best_tool_id in [DefaultTools.I_DONT_KNOW.value, DefaultTools.ANSWER_FROM_MEMORY.value]:

        if (best_tool_id == DefaultTools.I_DONT_KNOW.value):
            current_prompt = f"{bot_instructions}\n" if bot_instructions else "" \
                + (f"<{L_CONVERSATION}>{conversation_history}</{L_CONVERSATION}>\n" if len(conversation_history) else "") \
                + f"<{L_QUESTION}>{question}</{L_QUESTION}>\n" \
                + f"<{L_THOUGHT}>Maybe not enough information to answer. If that's the case, say why.</{L_THOUGHT}>\n" \
                + f"<{L_ANSWER}>"
        else:
            current_prompt = f"{bot_instructions}\n" if bot_instructions else "" \
                + (f"<{L_CONVERSATION}>{conversation_history}</{L_CONVERSATION}>\n" if len(conversation_history) else "") \
                + f"<{L_QUESTION}>{question}</{L_QUESTION}>\n" \
                + f"<{L_ANSWER}>"

        a, price = call_llm({
            "llm": LLMCallType.OPENAI_COMPLETION,
            "model": OpenAIModel.DAVINCI_TEXT.value,
            "max_tokens": 500,
            "prompt": current_prompt,
            "stop": f"</{L_ANSWER}>",
            "temperature": 0.1,
        })
        price_accumulator += price
        return a, [(question, a)], [], [], price_accumulator

    best_tool: SingleTool = get_tool_by_id(tools, best_tool_id)

    if "dynamic_params" not in best_tool:
        best_tool["dynamic_params"] = {}

    dynamic_params_to_fill = len(best_tool["dynamic_params"].keys())
    # only try to guess the "dynamic_params" if we have any
    if dynamic_params_to_fill == 0:
        tool_input = {}
    else:
        # Try more than once to get the right input.
        for attempt_to_find_input in range(__MAX_RETRIES_GUESS_INPUT):
            tool_input, price = __get_tool_input(
                best_tool,
                factual_memory,
                question,
                verbose,
                max_tokens=200,
                # higher temperature on subsequent attempts
                temperature=min(0.15 * attempt_to_find_input, 0.5)
            )
            price_accumulator += price

            filled_dynamic_params = len(tool_input.keys())
            if filled_dynamic_params == dynamic_params_to_fill:
                break

        # If, besides all attempts, we still don't have the right input...
        if filled_dynamic_params != dynamic_params_to_fill:
            raise Exception(
                f"Wrong input! Not enough parameters (actual: {filled_dynamic_params}, expected: {dynamic_params_to_fill})")

    try:

        response_instructions = question

        if "ai_response_prompt" in best_tool.keys():
            response_instructions = best_tool["ai_response_prompt"]

        answer, actual_url, raw_api_return, price = use_tool(
            best_tool,
            tool_input,
            question,
            factual_memory,
            verbose,
            bot_instructions,
            response_instructions
        )
        price_accumulator += price
    except Exception as e:
        print_big(e, "EXCEPTION CAUGHT! [#1]")
        cur_prompt_text = f"\n"\
            f"{bot_instructions}\n" \
            f"{conversation_history}\n" \
            f"--------\n" + \
            prompt_for_answer(question)

        a, price = call_llm({
            "llm": LLMCallType.OPENAI_COMPLETION,
            "model": OpenAIModel.DAVINCI_TEXT.value,
            "max_tokens": 500,
            "prompt": cur_prompt_text,
        })
        price_accumulator += price
        return a, [(question, a)], [], [], price_accumulator

    if verbose > 0:
        print_big("RESULT OF ANSWER")
        print_op("Question:  ", question)
        print_op("Tool Input:", tool_input)
        print_op("AI Answer: ", answer)

    return answer, \
        [(question, verbose_answer(raw_api_return, answer))], \
        [actual_url], \
        [{"response": "", "gpt_suggested_params": tool_input, "url": actual_url, "raw_api_return": raw_api_return}], \
        price_accumulator
