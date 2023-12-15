#!/usr/bin/python

from config.init import *

def handler(event, context):
    try:
        response_payload = {}
        if 'header' in event['params']:
            ## some function
            pass
        else:
            ## some function
            pass
        print(f"some_function: {some_function_results}")
        return(json.dumps(some_function_results))
    except Exception as e:
        print(f"Error in handler: {e}")
        return([])

if __name__ == "__main__":
    # remove json.dumpss()
    event_data = json.loads(json.dumps(sys.argv[1]))
    print("processing eventdata: ", event_data)
    event = {}
    event['Records'] = event_data
    context = None
    result = handler(event, context)
    if result['status'] != 200:
        exit(1)