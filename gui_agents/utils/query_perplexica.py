import requests
import json
import toml
import os

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)


def query_to_perplexica(query):
    # Load Your Port From the configuration file of Perplexica
    with open(
        os.path.join(os.path.dirname(parent_path), "Perplexica", "config.toml"), "r"
    ) as f:
        data = toml.load(f)
    port = data["GENERAL"]["PORT"]
    assert port, "You should set valid port in the config.toml"
    # Set the URL
    url = f"http://localhost:{port}/api/search"
    # Request Message
    message = {"focusMode": "webSearch", "query": query, "history": [["human", query]]}

    try:
        print("Sending Request to Perplexica...")
        response = requests.post(url, json=message)
    except requests.exceptions.RequestException as e:
        print("Error: Cannot connect to Perplexica due to the following error:", e)
        return ""
    if response.status_code == 200:
        return response.json()["message"]
    elif response.status_code == 400:
        raise ValueError(
            "The request is malformed or missing required fields, such as FocusModel or query"
        )
    else:
        raise ValueError("Internal Server Error")


# Test Code
if __name__ == "__main__":
    query = "What is Agent S?"
    response = query_to_perplexica(query)
    print(response)
