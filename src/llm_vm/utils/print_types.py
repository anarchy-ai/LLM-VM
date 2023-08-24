"""
This file contains a list of common functions across
the different agents that format and print output to the CLI.
"""

import sys


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
