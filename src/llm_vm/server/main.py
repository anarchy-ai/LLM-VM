import flask
from flask import request, jsonify
import json
import time
import traceback
import importlib
import openai
import os
import hashlib

from flask_cors import CORS
from contextlib import contextmanager


import llm_vm.server.routes as routes

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


app.register_blueprint(routes.bp)

def main_server_entry_point():
    # make this more configurable soon
    app.run(host="127.0.0.1", port=3002)

