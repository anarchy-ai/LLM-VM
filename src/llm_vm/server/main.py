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

from llm_vm.server import routes
app.register_blueprint(routes.bp)

def main_server_entry_point():
    # make this more configurable soon
    app.run(host="192.168.1.75", port=3002)

def generate_hash(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(input_string).encode('utf-8'))
    return int(sha256_hash.hexdigest(), 16) % 10**18

if __name__ == '__main__':
    app.run(port=3002)
