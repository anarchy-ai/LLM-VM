import re
from llm_vm.utils.keys import DICT_KEY_REGEX_TO_FIND_PURE_INTERPOLATIONS

###
# Use this regex to find if a variable is a pure interpolation.
# For @example, { "name": "{my_name}" } is pure, since there's nothing else besides the interpolation
# For @example, { "Ticker": "TickerID={ticker_id}|historical=1" } is NOT pure, since there is extra text
def __is_pure_interpolation (key: str) -> bool:
    return len(re.findall(DICT_KEY_REGEX_TO_FIND_PURE_INTERPOLATIONS, key)) == 1

###
# Use this to get the key of a pure interpolation ===> '{a}' => 'a', '{abc_def-345}' => 'abc_def-345'
def __get_pure_key (key: str) -> str:
    return key[1:-1]


def replace_variables_for_values(my_dict: dict, dynamic_keys: dict, ignore_key: str = "_______") -> any:
    def format_simple_value(value: str) -> any:
        try:
            if __is_pure_interpolation(value):
                formatted_value = dynamic_keys[__get_pure_key(value)]
            else:
                formatted_value = value.format(**dynamic_keys)
        except Exception as e:
            formatted_value = value

        return formatted_value

    replaced_dict = {}
    for key, value in my_dict.items():
        if (key == ignore_key):
            continue
        formatted_key = key.format(**dynamic_keys)
        if (isinstance(value, dict)):
            if (not isinstance(value, list) and not isinstance(value, dict)):
                formatted_value = format_simple_value(value)
            else:
                formatted_value = replace_variables_for_values(value, dynamic_keys, ignore_key)
        elif (isinstance(value, list)):
            formatted_value = []
            for item in value:
                if (not isinstance(item, list) and not isinstance(item, dict)):
                    formatted_value.append(format_simple_value(item))
                else:
                    formatted_value.append(replace_variables_for_values(item, dynamic_keys, ignore_key))
        else:
            formatted_value = format_simple_value(value)
        replaced_dict[formatted_key] = formatted_value
    return replaced_dict
