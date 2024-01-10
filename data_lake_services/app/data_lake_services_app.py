#!/usr/bin/python

#%%
from config.config_init import *

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
CORS(app)

# Swagger UI configuration
SWAGGER_URL = '/swagger'  # URL for exposing Swagger UI
API_URL = '/static/swagger.json'  # URL for the Swagger JSON

# Create a Swagger UI blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "GIS Data Lake Services"
    }
)

# Register the blueprint with Flask app
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/')
def home():
    return "<h1>Welcome to GIS Data Lake Services!</h1>"

@app.route('/items', methods=['GET'])
def get_items():
    # Example data - you can modify this with real data or database queries
    items = [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"},
        {"id": 3, "name": "Item 3"}
    ]
    return jsonify(items)

@app.route('/item', methods=['POST'])
def create_item():
    # Assuming JSON input with 'name' and 'description'
    data = request.json
    # Here you would typically process the data and save it to a database
    # For the sake of example, we'll just return it
    return jsonify({
        "status": "success",
        "data": data
    }), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
