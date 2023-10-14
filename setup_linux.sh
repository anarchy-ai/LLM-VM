#!/bin/bash

python -m venv llm
source llm/bin/activate
python -m pip install -e ."[dev]"
