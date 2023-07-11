import flask
from flask import request, jsonify
import json
import time
import traceback
import importlib
import openai
import os
import hashlib
# from llm_vm.completion.optimize import LocalOptimizer
# from test_agent import run_test
from flask_cors import CORS
from contextlib import contextmanager

# optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None)
# Flask Configuration
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

import llm_vm.server.routes as routes
app.register_blueprint(routes.bp)

def main_server_entry_point():
    # make this more configurable soon
    app.run(host="192.168.1.75", port=3002)

