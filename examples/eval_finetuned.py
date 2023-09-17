import sys
import re
import os
import pickle
from math import sqrt

import spacy

from llm_vm.client import Client

nlp = spacy.load("en_core_web_md")


class suppress_output:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        with open(os.devnull, "w", encoding='utf-8') as devnull:
            if self.suppress_stdout:
                self._stdout = sys.stdout
                sys.stdout = devnull

            if self.suppress_stderr:
                self._stderr = sys.stderr
                sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr


def squared_sum(x):
    """return 3 rounded square rooted value"""

    return round(sqrt(sum(a * a for a in x)), 3)


# metrics that can be used for evaluation


def cos_similarity(x, y):
    """return cosine similarity between two lists"""

    numerator = sum(a * b for a, b in zip(x, y))
    denominator = squared_sum(x) * squared_sum(y)
    return round(numerator / float(denominator), 3)


def regex_check(string, regex):
    if re.match(regex, string) is not None:
        return 1
    return 0


def metric():
    return regex_check


with open("data_gen.pkl", "rb") as new_file:
    examples = list(pickle.load(new_file))
with suppress_output(suppress_stdout=True, suppress_stderr=True):
    client_test = Client(big_model='pythia')
    # specify the file name of the finetuned model to load
    model_name = '2023-08-22T17-31-20_pythia-70m-deduped.pt'
    client_test.load_finetune(model_name)
metrics = []
for i in examples:
    with suppress_output(suppress_stdout=True, suppress_stderr=True):
        response_test = client_test.complete(prompt=i[0], context='')[
            "completion"].split("<END>")[0]
    ground_truth = i[1].split("<END>")[0]
    print("Response: " + response_test, "Ground Truth: " + ground_truth)
    # final_met = [metric()(nlp(response_test).vector, nlp(ground_truth).vector)]
    final_met = [
        metric()(response_test, r"\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)")]
    print(final_met)
    metrics.append(final_met)
print(metrics)
