from flask_caching import Cache
from flask import request

def make_cache_key(*args, **kwargs):
    key = request.url + str(request.data)
    return key

config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 30,
    'CACHE_KEY_PREFIX': 'custom_prefix',
    'CACHE_KEY_FUNC': make_cache_key
}

cache = Cache(config=config)