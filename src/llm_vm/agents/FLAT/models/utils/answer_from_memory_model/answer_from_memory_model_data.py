from llm_vm.utils.typings_llm import *

answer_from_memory_data: AnswerInMemoryModel = {
    "openai_model": OpenAIModel.DAVINCI,
    "data": [
        {
            "question": "What's the time?",
            "answer": False
        },
        {
            "mem": [
                ("Whats day is it today?", "Today is Sunday, December 15th, 2027. The time is 11:33AM")
            ],
            "question": "What's the time?",
            "answer": True
        },
        {
            "question": "How are you feeling?",
            "answer": True
        },
        {
            "question": "What color is the sky?",
            "answer": True
        },
        {
            "question": "What is the temperature in Portland?",
            "answer": False
        },
        {
            "mem": [
               ("What is the weather in Portland?", "It is sunny, 22C, with clear skies. Theres 40% chance of rain after 18:00hs")
            ],
            "question": "What is the temperature in Portland?",
            "answer": True
        },
        {
            "question": "What day is it today?",
            "answer": False
        },
        {
            "question": "What's the name of this place?",
            "answer": False
        }
    ]
}
