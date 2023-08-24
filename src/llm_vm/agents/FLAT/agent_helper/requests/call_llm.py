from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.agent_helper.requests.call_open_ai import call_open_ai

def call_llm(llm_request: LLMCallParams) -> LLMCallReturnType:
    if llm_request['llm'] == LLMCallType.OPENAI_CHAT or llm_request['llm'] == LLMCallType.OPENAI_COMPLETION:
        try:
            return call_open_ai(llm_request)
        except Exception as e:
            print("OpenAI call failed", e, file=sys.stderr)
            return "OpenAI is down!", 0

