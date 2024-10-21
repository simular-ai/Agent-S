import requests
import json

def query_to_perplexica(query):
    # Your URL for searching
    url = 'http://localhost:3001/api/search'
    
    # Your request
    message = {
        "chatModel": {
            "provider": "openai",
            "model": "gpt-4o-mini"
        },
        "embeddingModel": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        },
        "focusMode": "webSearch",
        "query": query,
        "history": [
            ["human", query]
        ]
    }

    response = requests.post(url, json=message)

    if response.status_code == 200:
        return response.json()['message'] 
    elif response.status_code == 400:
        raise ValueError('The request is malformed or missing required fields, such as FocusModel or query')
    else:
        raise ValueError('Internal Server Error')
    
# For test of availability of Perplexica, simply run this script
if __name__ == "__main__":
    query = "What is Agent S?"
    response = query_to_perplexica(query)
    print(response)
