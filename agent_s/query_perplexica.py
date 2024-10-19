import requests
import json

def query_to_perplexica(query):
    url = 'http://localhost:3001/api/search'
    
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
    # print(response)
    if response.status_code == 200:
        return response.json()['message']
    elif response.status_code == 400:
        raise ValueError('The request is malformed or missing required fields, such as FocusModel or query')
    else:
        raise ValueError('Internal Server Error')
    

if __name__ == "__main__":
    query = "What is Agent S?"
    response = query_to_perplexica(query)
    print(response)
