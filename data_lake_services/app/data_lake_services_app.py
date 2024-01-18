#!/usr/bin/python

#%%
from config.config_init import *

import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
logger = logging.getLogger('werkzeug')  # This is the Flask logger
logger.setLevel(logging.INFO)

# Setting the UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GIS Data Lake Services</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }
            .container { max-width: 1200px; margin: auto; padding: 20px; }
            .header { background: #005792; padding: 20px 0; color: white; text-align: center; }
            .header h1 { margin: 0; }
            .content { background: white; padding: 20px; margin-top: 20px; }
            .content h2 { color: #005792; }
            .footer { text-align: center; margin-top: 40px; padding: 20px 0; font-size: 0.8em; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Welcome to GIS Data Lake Services</h1>
            </div>
            <div class="content">
                <h2>About Our Services</h2>
                <p>The GIS Data Lake Services API provides comprehensive access to the data in the GIS data lake, encompassing a wide range of functionalities. These include data ingestion, advanced querying capabilities, analytical processing, and secure data removal when necessary.</p>
                
                <h2>Security and Standards</h2>
                <p>Security is our utmost priority. The API adheres to modern security standards, ensuring that your data is protected against unauthorized access and nefarious activities. We employ custom encryption methods, alongside industry-standard protocols, to safeguard your data throughout its lifecycle in the data lake.</p>

                <h2>Best Practices</h2>
                <p>Our API is designed with best-in-class software practices, ensuring reliability, scalability, and maintainability. We continuously update our services to reflect the latest trends in technology, providing you with a cutting-edge data management solution.</p>
            </div>
            <div class="footer">
                GIS Data Lake Services © 2024. All rights reserved.
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/search', methods=['POST'])
def perform_search():
    logger.info("Received search request")

    query_str = request.json.get('query') if request.json else None

    if query_str is None:
        logger.warning("No query provided in search request")
        return jsonify({"status": "error", "message": "No query provided"}), 400

    # Here you would typically perform a search query against a database
    try:
        results = search(query_str)
        logger.info(f"Search performed successfully for query: {query_str}")
        return jsonify({
            "status": "success",
            "data": results
        }), 200
    except Exception as e:
        logger.error(f"Error during search operation: {e}")
        return jsonify({"status": "error", "message": "Search operation failed"}), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    # Check if the post request has the file part
    if 'document' not in request.files:
        logger.error("No file part in the request")
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['document']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        logger.error("No file selected for upload")
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        document_name = request.form.get('name', filename)  # Default to original filename if name not provided

        # Save the file to the specified UPLOAD_FOLDER
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Output the file type
        file_type = file.content_type
        logger.info(f"File '{filename}' uploaded successfully as '{document_name}'")
        logger.info(f"Saved at '{file_path}' with file type '{file_type}'")

    return jsonify({
        "status": "success",
        "message": f"File '{document_name}' uploaded successfully",
        "file_path": file_path,
        "file_type": file_type
    }), 200

########################################
####    RabbitMQ Process Message    ####
########################################

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Start a new event loop in a background thread
loop = asyncio.new_event_loop()
t = threading.Thread(target=start_loop, args=(loop,))
t.start()

@app.route('/send', methods=['POST'])
def send():
    logger.info("Received POST request on '/send'")
    data = request.json

    if data and 'message' in data:
        logger.info("Valid request data received. Preparing to send message.")
        try:
            # Assuming 'send_message' is an asyncio coroutine and 'loop' is the asyncio event loop
            asyncio.run_coroutine_threadsafe(send_message('example_queue', data['message']), loop)
            logger.info(f"Message sent to queue: {data['message']}")
            return jsonify({"status": "Message sent"}), 200
        except Exception as e:
            logger.error(f"Error in sending message: {e}")
            return jsonify({"status": "error", "message": "Failed to send message"}), 500
    else:
        logger.warning("Invalid request: No message found in request data")
        return jsonify({"error": "Invalid request"}), 400

@app.route('/start_consumer', methods=['GET'])
def start_consumer():
    logger.info("Starting consumer on 'example_queue'")
    try:
        asyncio.run_coroutine_threadsafe(consume_message('example_queue', process_message), loop)
        logger.info("Consumer successfully started")
        return "Consumer started", 200
    except Exception as e:
        logger.error(f"Error starting consumer: {e}")
        return "Error starting consumer", 500

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
