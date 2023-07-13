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
import argparse, configparser

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", type=str, help='Config File')
parser.add_argument("-f", "--foo", type=int, default=5, help='Foo Number. Default: 5')

# Initialize Flask App
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
        server_entry_point(host = host, port = port)
    else:
        server_entry_point()

    
if __name__ == '__main__':
    cli()
