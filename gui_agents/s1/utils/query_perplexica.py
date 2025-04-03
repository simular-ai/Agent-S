import requests
import toml
import os


def query_to_perplexica(query):
    # Retrieve the URL from an environment variable
    url = os.getenv("PERPLEXICA_URL")
    if not url:
        raise ValueError(
            "PERPLEXICA_URL environment variable not set. It may take the form: 'http://localhost:{port}/api/search'. The port number is set in the config.toml in the Perplexica directory."
        )

    # Request Message
    message = {"focusMode": "webSearch", "query": query, "history": [["human", query]]}

    response = requests.post(url, json=message)

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
