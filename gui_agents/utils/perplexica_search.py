import os
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import re
import requests
import time
from bs4 import BeautifulSoup


def perplexica_search(query):
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    try:
        current_dict = json.load(
            open((os.path.join(current_dir, "perplexica_search.json")))
        )
    except:
        current_dict = {}
    if query in current_dict.keys():
        return current_dict[query]

    result = ""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    search_url = f"http://localhost:3000/?q={query}"
    driver.get(search_url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "li"))
        )
        time.sleep(3)
    except:
        time.sleep(20)

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, "html.parser")
    li_results = soup.find_all("li")
    for li in li_results:
        text = li.get_text(separator=" ", strip=True)
        text = re.sub(r" \d+( \d+)* \.", ".", text)
        text = re.sub(r"\. \d+( \d+)*", ".", text)
        text = re.sub(r"\" \d+( \d+)*", '"', text)
        result = result + text + "\n"

    driver.quit()

    current_dict[query] = result.strip()
    with open(os.path.join(current_dir, "perplexica_search.json"), "w") as fout:
        json.dump(current_dict, fout, indent=2)

    return result.strip()


def _test_search():

    queries = [
        "How to change slide background color to purple in LibreOffice Impress on Ubuntu and add title to notes?"
    ]

    for q in queries:

        res = perplexica_search(q)
        print(res)


if __name__ == "__main__":
    _test_search()
