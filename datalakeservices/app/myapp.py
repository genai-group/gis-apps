#!/usr/bin/python

from config.init import *


from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/item', methods=['POST'])
def create_item():
    data = request.json
    return jsonify({"message": "Item created", "data": data}), 201

@app.route('/api/item/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    data = request.json
    return jsonify({"message": f"Item {item_id} updated", "data": data}), 200

@app.route('/api/item/<int:item_id>', methods=['GET'])
def get_item(item_id):
    # Example: Fetch item details from your data source
    item = {"id": item_id, "name": "Example Item", "description": "This is an example."}
    return jsonify(item), 200

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, jsonify, request

# app = Flask(__name__)

# @app.route('/some_endpoint', methods=['GET', 'POST'])
# def some_function():
#     try:
#         # Your logic here
#         # Use request.args or request.json to access request data
#         return jsonify({"message": "Success"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)



# def handler(event, context):
#     try:
#         response_payload = {}
#         if 'header' in event['params']:
#             ## some function
#             pass
#         else:
#             ## some function
#             pass
#         print(f"some_function: {some_function_results}")
#         return(json.dumps(some_function_results))
#     except Exception as e:
#         print(f"Error in handler: {e}")
#         return([])

# if __name__ == "__main__":
#     # remove json.dumpss()
#     event_data = json.loads(json.dumps(sys.argv[1]))
#     print("processing eventdata: ", event_data)
#     event = {}
#     event['Records'] = event_data
#     context = None
#     result = handler(event, context)
#     if result['status'] != 200:
#         exit(1)