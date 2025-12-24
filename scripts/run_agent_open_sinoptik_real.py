"""Open sinoptik.ua Odessa page in browser (real actions), then fetch and parse monthly forecast and save a screenshot.

Actions performed:
- Open the Odessa page in your default browser (usually Chrome).
- Wait a few seconds for the page to load.
- Save a full-screen screenshot to results/sinoptik_odessa.png.
- Fetch the same URL and attempt to extract a nearby block containing "Погода на месяц" or similar and print it.

CAUTION: this script will interact with your desktop (open browser) and take a screenshot.
"""
import os
import time
import webbrowser
import urllib.request
import urllib.error
from urllib.parse import quote
from html import unescape
from PIL import Image
import pyautogui

URL_CANDIDATES = [
    'https://sinoptik.ua/погода-одесса/',
    'https://sinoptik.ua/пogoda-odessa/',
    'https://sinoptik.ua/odessa/',
]

os.makedirs('results', exist_ok=True)

class RealOpenSinoptikAgent:
    def __init__(self):
        self.called = False

    def reset(self):
        self.called = False

    def predict(self, instruction: str, observation: dict):
        if not self.called:
            self.called = True
            # Open first candidate URL in default browser
            code = "import webbrowser, time; webbrowser.open(\'https://sinoptik.ua/погода-одесса/\'); time.sleep(1)"
            return {"reflection": "open sinoptik"}, [code, "done"]
        else:
            return {"reflection": "done"}, ["done"]


def fetch_month_block(url):
    tried = []
    for u in URL_CANDIDATES:
        try:
            # Ensure proper quoting for non-ascii parts
            u_quoted = u if all(ord(c) < 128 for c in u) else quote(u, safe=':/')
            resp = urllib.request.urlopen(u_quoted, timeout=10)
            html = resp.read().decode('utf-8', errors='replace')
            # Search for keywords
            keywords = ['Погода на месяц', 'Погода на місяць', 'Месяц', 'на месяц', 'На місяць']
            for kw in keywords:
                idx = html.find(kw)
                if idx != -1:
                    start = max(0, idx-800)
                    end = min(len(html), idx+2400)
                    block = unescape(html[start:end])
                    return (u, kw, block)
            tried.append((u, 'no-keyword'))
        except Exception as e:
            tried.append((u, str(e)))
    return (None, None, tried)


def main():
    # Use the agent to open the page (real browser action)
    agent = RealOpenSinoptikAgent()
    scaled_w, scaled_h = 320, 180
    print('Opening sinoptik.ua in default browser...')
    # run_agent will execute the webbrowser.open code once
    from gui_agents.s3 import cli_app
    cli_app.run_agent(agent, 'Open sinoptik Odessa page', scaled_w, scaled_h, require_exec_confirmation=False)

    # Wait for page to load visually
    print('Waiting a few seconds for the browser to load the page...')
    time.sleep(4)

    # Save screenshot
    screenshot_path = os.path.join('results', 'sinoptik_odessa.png')
    img = pyautogui.screenshot()
    img.save(screenshot_path)
    print('Saved screenshot to', screenshot_path)

    # Fetch and try to extract month block
    print('Fetching page content and looking for "Погода на месяц" block...')
    url, kw, block = fetch_month_block(URL_CANDIDATES)
    if url:
        print('Found keyword', kw, 'on', url)
        # Save block to file
        with open(os.path.join('results', 'sinoptik_odessa_block.html'), 'w', encoding='utf-8') as f:
            f.write(block)
        print('Saved extracted HTML block to results/sinoptik_odessa_block.html')
        # Print small preview
        preview = block.replace('\n', ' ')[:1000]
        print('Preview:', preview)
    else:
        print('Could not find month block; attempts:', block)


if __name__ == '__main__':
    main()
