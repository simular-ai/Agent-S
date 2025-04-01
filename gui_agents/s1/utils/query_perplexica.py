import requests
import toml
import os


def query_to_perplexica(query):
    config_path = os.getenv("PERPLEXICA_CONFIG_PATH")
    if not config_path:
        raise ValueError(
            "Environment variable PERPLEXICA_CONFIG_PATH is not set. Please set it to the path of your config.toml."
        )

    # Load Your Port From the configuration file of Perplexica
    with open(config_path, "r") as f:
        data = toml.load(f)
    port = data["GENERAL"]["PORT"]
    assert port, "You should set valid port in the config.toml"

    # Retrieve the URL from an environment variable
    url = os.getenv("PERPLEXICA_URL")
    if not url:
        url = f"http://localhost:{port}/api/search"
        print(
            f"PERPLEXICA_URL not set, using default URL: http://localhost:{port}/api/search"
        )

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
