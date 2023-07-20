from llm_vm.utils.typings_llm import *

# Retuns the tools, in a random order, and with randomised IDs.
def get_randomised_training_tools (
    generic_tools: ToolList,
    shuffled_by: int = 0,
    shuffle_by_modulo: int = 1e9
) -> ToolList:

    def __shuffled(tool_id: int) -> int:
        return int((tool_id + shuffled_by) % shuffle_by_modulo)

    shuffled_generic_tools: ToolList = [{"id": __shuffled(t["id"]), "description": t["description"]} for t in generic_tools]

    tools: ToolList = shuffled_generic_tools + [
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_SEND_EMAIL.value),
            "description": "Use this tool to send an email to someone"
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_BOOK_TABLE.value),
            "description": "Useful to book a table or slot at a restaurant"
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_CREATE_DOCUMENT.value),
            "description": "Create a document on google drive"
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_SHARE_DOCUMENT.value),
            "description": "Useful to share an online document with someone. You need the email"
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_FIND_RESTAURANT.value),
            "description": "Use this tool to find a specific bar, café or restaurant, or to search for places to eat around a certain location."
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_NEWS.value),
            "description": "Use this tool to get news and current events"
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_BOOK_APPOINTMENT.value),
            "description": "Use this tool to book an appointment, meeting or doctor, through a calendar."
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_AVAILABLE_APPOINTMENT.value),
            "description": "Use this tool to get the available time slots for meetings, doctors, appointments, etc."
        },
        {
            "id": __shuffled(DefaultTrainingTools.TRAIN_ORDER_FOOD.value),
            "description": "Use this tool to order food and get it delivered."
        }
    ]

    return tools
