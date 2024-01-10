#!/usr/bin/python

#%%
from config.init import *

from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint

#%%
app = Flask(__name__)

# Swagger UI configuration
SWAGGER_URL = '/swagger'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Our API url (can be a local resource or remote)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "GIS Data Lake Services"
    }
)

@app.route('/')
def home():
    # This route will display a simple message or HTML page
    return "<h1>Welcome to the Flask API!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
