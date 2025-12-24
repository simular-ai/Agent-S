"""Run Agent-S to create a single-page restaurant website and open it in Google Chrome.

This script uses a small agent that returns code which writes HTML/CSS/JS to
results/restaurant/index.html and attempts to open it in Chrome. It executes
that code once (real actions), then stops.

CAUTION: This will create files in the repo and open a browser window.
"""
import os
import platform
from PIL import Image
import pyautogui

from gui_agents.s3 import cli_app


def _default_chrome_paths():
    paths = []
    if platform.system().lower() == "windows":
        paths.extend([
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ])
    elif platform.system().lower() == "darwin":
        # macOS
        paths.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    else:
        # common Linux paths
        paths.extend([
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/snap/bin/chromium",
        ])
    return paths


class RealCreateSiteAgent:
    def __init__(self):
        self.called = False

    def reset(self):
        self.called = False

    def predict(self, instruction: str, observation: dict):
        if not self.called:
            self.called = True
            print('RealCreateSiteAgent.predict: returning site-creation code')
            code = r"""
import os, subprocess, webbrowser
html = '''<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ресторан "У моря"</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; color: #333 }
    header { background: linear-gradient(90deg,#0066cc,#33aaff); color: #fff; padding: 3rem 1rem; text-align:center }
    .container { max-width: 900px; margin: 1rem auto; padding: 1rem }
    .hero { display:flex; gap: 1rem; align-items:center }
    .hero img { width:50%; border-radius:8px }
    .menu { display:flex; gap:1rem; flex-wrap:wrap }
    .menu-item { flex:1 1 200px; border:1px solid #eee; padding:1rem; border-radius:6px }
    footer { text-align:center; padding:1rem; color:#666 }
    @media (max-width:600px){ .hero{flex-direction:column} .hero img{width:100%} }
  </style>
</head>
<body>
<header>
  <h1>Ресторан "У моря"</h1>
  <p>Свежая еда, уютная атмосфера и вид на море</p>
</header>
<main class="container">
  <section class="hero">
    <img src="https://images.unsplash.com/photo-1498654896293-37aacf113fd9?w=800&q=80" alt="restaurant">
    <div>
      <h2>Добро пожаловать!</h2>
      <p>Мы предлагаем блюда из местных продуктов, авторские закуски и широкий выбор напитков.</p>
      <p><strong>Часы работы:</strong> 10:00 — 23:00</p>
    </div>
  </section>

  <section>
    <h3>Меню</h3>
    <div class="menu">
      <div class="menu-item"><h4>Салат из лосося</h4><p>Свежий лосось, микс салатов, цитрусовая заправка — 320 ₴</p></div>
      <div class="menu-item"><h4>Паста морская</h4><p>Паста с морепродуктами в сливочном соусе — 380 ₴</p></div>
      <div class="menu-item"><h4>Стейк</h4><p>Говяжий стейк с овощами — 420 ₴</p></div>
    </div>
  </section>

  <section>
    <h3>Контакты</h3>
    <p>Адрес: Одесса, набережная, 1</p>
    <p>Телефон: +38 0XX XXX XX XX</p>
  </section>
</main>
<footer>
  © Ресторан "У моря" — Приятного аппетита!
</footer>
</body>
</html>'''

out_dir = os.path.join('results','restaurant')
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir,'index.html')
with open(path,'w',encoding='utf-8') as f:
    f.write(html)

# Try to open in Google Chrome if available
paths = []
# Platform-aware chrome candidates
import sys
if sys.platform.startswith('win'):
    paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
elif sys.platform == 'darwin':
    paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
else:
    paths = ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/snap/bin/chromium"]

chrome_path = None
for p in paths:
    if os.path.exists(p):
        chrome_path = p
        break

try:
    if chrome_path:
        subprocess.Popen([chrome_path, os.path.abspath(path)])
    else:
        webbrowser.open('file://' + os.path.abspath(path))
    print('Opened site in browser')
except Exception as e:
    print('Failed to open browser:', e)

"""

            return {"reflection": "created restaurant site"}, [code, "done"]
        else:
            return {"reflection": "done"}, ["done"]


def main():
    agent = RealCreateSiteAgent()
    scaled_w, scaled_h = 320, 180
    print('Creating single-page restaurant website and attempting to open it in Chrome...')
    cli_app.run_agent(agent, 'Create a single-page restaurant website and open it in Chrome', scaled_w, scaled_h, require_exec_confirmation=False)

    # Give the browser a moment to open
    import time
    time.sleep(2)

    out_path = os.path.join('results', 'restaurant', 'index.html')
    if os.path.exists(out_path):
        print('Site created at', out_path)
        # Try to capture a screenshot of the desktop for evidence
        try:
            img = pyautogui.screenshot()
            os.makedirs('results', exist_ok=True)
            shot_path = os.path.join('results', 'restaurant_screenshot.png')
            img.save(shot_path)
            print('Saved desktop screenshot to', shot_path)
        except Exception as e:
            print('Could not take screenshot:', e)
    else:
        print('Site was not created; check agent output for errors')


if __name__ == '__main__':
    main()
