"""
This file contains various machine learning classes and variables
for training and API calls to OpenAI.
"""

from typing import Tuple, List, Dict, TypedDict, Union, Any, Optional
from enum import Enum

class OpenAIModel(Enum):
    CURIE = "curie"
    CURIE_TEXT = "text-curie-001"
    FAST_DAVINCI = "code-cushman-002"
    DAVINCI = "davinci"
    DAVINCI_TEXT = "text-davinci-003"

class LLMCallType(Enum):
    OPENAI_COMPLETION = "openai-completion"
    OPENAI_CHAT = "openai-chat"
    # ... other models

class DecisionStep(Enum):
    SPLIT = "split_questions"
    INPUT = "guess_api_input"
    FROM_MEMORY = "use_memory"
    TOOL_PICKER = "tool_picker"

class DefaultTools(Enum):
    I_DONT_KNOW = -1
    ANSWER_FROM_MEMORY = 0

    WOLFRAM = 1
    DIRECTIONS = 2
    WEATHER = 3

    ## Always set this to the latest ID. This is an integer, not an enum!
    __LAST_ID__: int = 4

class DefaultTrainingTools(Enum):
    TRAIN_SEND_EMAIL = 4
    TRAIN_CREATE_DOCUMENT = 5
    TRAIN_SHARE_DOCUMENT = 6
    TRAIN_BOOK_TABLE = 7
    TRAIN_FIND_RESTAURANT = 8
    TRAIN_NEWS = 9
    TRAIN_ORDER_FOOD = 10
    TRAIN_BOOK_APPOINTMENT = 11
    TRAIN_AVAILABLE_APPOINTMENT = 12
    TRAIN_FIND_FLIGHT = 13

TupleList = List[Tuple[str, str]]

class ToolTypeArgs(TypedDict):
    url: str
    params: Optional[dict]
    json: Optional[dict]
    jsonParams: Optional[dict]
    auth: Optional[dict]
    cert: Optional[str]
    data: Optional[dict]

class SingleTool(TypedDict):
    description: str
    dynamic_params: dict
    id: int
    args: ToolTypeArgs
    ai_response_prompt: Union[str, None]

ToolList = List[SingleTool]
# ToolList = NewType("ToolList", List[SingleTool])
EnumeratedToolList = List[Tuple[int,SingleTool]]

ALL_TOOLS_MAP: Dict[str, Union[DefaultTools, DefaultTrainingTools]] = {
    "dont_know": DefaultTools.I_DONT_KNOW,
    "memory": DefaultTools.ANSWER_FROM_MEMORY,
    "weather": DefaultTools.WEATHER,
    "wolfram": DefaultTools.WOLFRAM,
    "directions": DefaultTools.DIRECTIONS,
    "send_email": DefaultTrainingTools.TRAIN_SEND_EMAIL,
    "create_doc": DefaultTrainingTools.TRAIN_CREATE_DOCUMENT,
    "share_doc": DefaultTrainingTools.TRAIN_SHARE_DOCUMENT,
    "book_table": DefaultTrainingTools.TRAIN_BOOK_TABLE,
    "find_restaurant": DefaultTrainingTools.TRAIN_FIND_RESTAURANT,
    "news": DefaultTrainingTools.TRAIN_NEWS,
    "order_food": DefaultTrainingTools.TRAIN_ORDER_FOOD,
    "book_appointment": DefaultTrainingTools.TRAIN_BOOK_APPOINTMENT,
    "check_appointment": DefaultTrainingTools.TRAIN_AVAILABLE_APPOINTMENT,
    "find_flight": DefaultTrainingTools.TRAIN_FIND_FLIGHT
}

class QuestionSplitModelJSONData(TypedDict):
    mem: List[List[str]]
    question: str
    answer: List[str]

class QuestionSplitModelData(TypedDict):
    mem: Union[TupleList, None]
    question: str
    answer: Union[List[str], None]
    tools: Union[ToolList, None]

class AnswerFromMemoryModelData(TypedDict):
    mem: Union[TupleList, None]
    question: str
    thought: Union[str, None]
    answer: Union[str, None]

class QuestionSplitInputModel(TypedDict):
    model_name: str
    openai_model: OpenAIModel
    data: List[QuestionSplitModelData]

class PromptModelEntry(TypedDict):
    prompt: str
    completion: str

class LLMCallParams(TypedDict):
    llm: LLMCallType
    model: Tuple[str, Union[str, None]]
    prompt: str
    temperature: float
    max_tokens: int
    stop: str

LLMCallReturnType = Tuple[str, float]

class LLMTrainingResult(TypedDict):
    elapsed_time_s: int
    model_name: str
    model_files: List[str]

class LLMTrainingResultMeta(TypedDict):
    OPENAI_KEY: str
    DATE: str
    TIMESTAMP: int

LLModels = Union[Dict[str, LLMTrainingResult], LLMTrainingResultMeta]


class ToolInputModelJSONData(TypedDict):
    mem: Optional[List[str]]
    description: str
    question: str
    answer: dict
    params: dict
class ToolInputModelData(TypedDict):
    mem: Optional[TupleList]
    description: str
    question: str
    answer: dict
    params: dict

class ToolInputModel(TypedDict):
    openai_model: OpenAIModel
    data: List[ToolInputModelData]

class ToolpickerInputModelJSONData(TypedDict):
    mem: Optional[List[str]]
    thought: Optional[str]
    question: str
    answer: str
class ToolpickerInputModelData(TypedDict):
    mem: Optional[TupleList]
    thought: Optional[str]
    question: str
    answer: int
    tools: Optional[ToolList]

class ToolpickerInputModel(TypedDict):
    openai_model: OpenAIModel
    data: List[ToolpickerInputModelData]

class AnswerInMemoryModelData(TypedDict):
    mem: Optional[TupleList]
    facts: Optional[TupleList]
    question: str
    answer: Optional[bool]

class AnswerInMemoryModel(TypedDict):
    openai_model: OpenAIModel
    data: List[AnswerInMemoryModelData]

class DebugCalls(TypedDict):
    response: str
    gpt_suggested_params: Any
    url: str
    raw_api_return: str

DebugCallList = List[DebugCalls]
