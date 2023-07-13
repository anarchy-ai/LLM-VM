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
from llm_vm.config import args

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

# Register_blueprint from routes to load API
app.register_blueprint(routes.bp)

def server_entry_point(host = '127.0.0.1', port = 3002):
    """
    This function launches the server with specified parameters

     Parameters:
         host (str): Network IP address
         port (int): Port Number

     Returns:
         None

     Example:
         >>> server_entry_point(port = 3002)
    """
    app.run(host = host,port = port)
    
def cli():
    """
     This function is the entry point for the project and allows the user to specify an option network address and port number when launching from the cli

     Parameters:
         None

     Returns:
         None
     
    """
    port = args.port
    if port > 65535:
        print('Port defined out of range, defaulting to 3002')
        port = 3002
    server_entry_point(host = args.host, port = port)    
    
if __name__ == '__main__':
    cli()
