from llm_vm.server import app

import flask
from flask import request, jsonify
import json
import time
import traceback
import importlib
import openai
import os
import hashlib

# from test_agent import run_test
from flask_cors import CORS
from contextlib import contextmanager

import llm_vm.server.routes as routes 
from llm_vm.completion.optimize import LocalOptimizer

optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None)
# Flask Configuration
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


app.register_blueprint(routes.bp)


# we are gonna want 
def main_server_entry_point():
    # make this more configurable soon
    app.run(host="192.168.1.75", port=3002)


# we can reenable this if needed, but really use the cli_entry-points in pyproject.toml please
# if __name__ == '__main__':
#     # app.run(host="192.168.1.75",port=3002) # run at specified IP
#     main_server_entry_point() # for running local
