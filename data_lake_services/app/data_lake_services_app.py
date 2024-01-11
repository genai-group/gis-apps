#!/usr/bin/python

#%%
from config.config_init import *

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
CORS(app)

# Celery
if GIS_ENVIRONMENT == 'local':
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
elif GIS_ENVIRONMENT == 'flask-local':
    app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'
else:
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Swagger UI configuration
SWAGGER_URL = '/gis-data-lake-services'  # URL for exposing Swagger UI
API_URL = '/static/gis-data-lake-services.json'  # URL for the Swagger JSON

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

@app.route('/search', methods=['POST'])
def perform_search():
    query_str = request.json.get('query') if request.json else None

    if query_str is None:
        return jsonify({"status": "error", "message": "No query provided"}), 400

    # Here you would typically perform a search query against a database
    results = search(query_str)  # Assuming 'search' is a defined function

    # For the sake of example, we'll just return the query
    return jsonify({
        "status": "success",
        "data": results
    }), 200

# @app.route('/search', methods=['POST'])
# def perform_search():
#     data = request.json.get('query')
#     if not data:
#         return jsonify({"error": "No search query provided"}), 400
#     task = process_search_data.delay(data)
#     return jsonify({"task_id": task.id}), 202

# @celery.task
# def process_search_data(query):
#     # Call the original search function here
#     results = original_search_function(query)
#     return results

# @app.route('/status/<task_id>', methods=['GET'])
# def get_task_status(task_id):
#     task = process_search_data.AsyncResult(task_id)
#     if not task:
#         return jsonify({"error": "Invalid task ID"}), 404
#     return jsonify({"task_id": task_id, "status": task.status})

# @app.route('/result/<task_id>', methods=['GET'])
# def get_task_result(task_id):
#     task = process_search_data.AsyncResult(task_id)
#     if not task:
#         return jsonify({"error": "Invalid task ID"}), 404
#     if task.status == 'SUCCESS':
#         return jsonify({"task_id": task_id, "status": task.status, "result": task.result})
#     return jsonify({"task_id": task_id, "status": task.status}), 202

# def original_search_function(query):
#     # Implement your actual search logic here
#     results = search(query)
#     time.sleep(2)  # Simulating a time-consuming search operation
#     return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
