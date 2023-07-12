import flask
from flask import request, jsonify
import json
import time
import traceback
import importlib
import openai
import os
import hashlib
import sys

from flask_cors import CORS
from contextlib import contextmanager


import llm_vm.server.routes as routes

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


app.register_blueprint(routes.bp)

def main_server_entry_point(host = '127.0.0.1', port = 3002):
    # make this more configurable soon
    app.run(host = host,port = port)

def cli():
    if len(sys.argv) > 1:
        try:
            # We're going to except on a value error at the int exchange
            # If the cli argument isn't an integer then we'll try it as an address
            port = int(sys.argv[1])
            host = sys.argv[2] if len(sys.argv) > 2 else '127.0.0.1'
        except:
            # if it doesn't fit the above then we're likely looking at an addres
            host = sys.argv[1]
            # check if the port has been specified
            port = int(sys.argv[2]) if len(sys.argv) > 2 and int(sys.argv[2]) < 65536 else 3002
        if port > 66535:
            print('Port defined out of range, defaulting to 3002')
            port = 3002
        main_server_entry_point(host = host, port = port)
    else:
        main_server_entry_point()

    
if __name__ == '__main__':
    cli()
