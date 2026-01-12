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
    :root { --accent: #ff6b6b; --muted: rgba(255,255,255,0.85) }
    html,body { height:100%; margin:0 }
    body {
      font-family: Arial, Helvetica, sans-serif;
      color: var(--muted);
      background: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3?w=1600&q=80&auto=format&fit=crop') center/cover fixed no-repeat;
      -webkit-font-smoothing:antialiased;
      -moz-osx-font-smoothing:grayscale;
    }
    .overlay { background: rgba(0,0,0,0.38); min-height:100%; }
    header { display:flex; align-items:center; justify-content:space-between; padding:1rem 2rem; }
    header h1{ margin:0; font-size:1.6rem }
    nav a{ color:var(--muted); margin-left:1rem; text-decoration:none; font-weight:600 }

    .hero { display:flex; gap:2rem; align-items:center; padding:6rem 2rem; max-width:1200px; margin:0 auto }
    .hero .intro{ max-width:640px }
    .hero h2{ font-size:2.4rem; margin:0 0 .5rem; text-shadow:0 6px 24px rgba(0,0,0,0.6) }
    .hero p{ margin:.5rem 0 1rem; line-height:1.45 }
    .btn{ display:inline-block; background:var(--accent); color:#fff; padding:.75rem 1.1rem; border-radius:999px; text-decoration:none; font-weight:700 }

    .section{ padding:3rem 2rem; max-width:1100px; margin:0 auto }
    .menu{ display:flex; gap:1rem; flex-wrap:wrap }
    .menu-item{ background:rgba(255,255,255,0.04); padding:1rem; border-radius:10px; flex:1 1 240px; transition:transform .18s ease, box-shadow .18s ease }
    .menu-item:hover{ transform:translateY(-8px) scale(1.02); box-shadow:0 12px 30px rgba(0,0,0,0.5) }
    .menu-item h4{ margin:.2rem 0 }

    footer{ text-align:center; padding:2rem; color:rgba(255,255,255,0.8) }

    /* small screens */
    @media (max-width:700px){
      .hero{ padding:3rem 1rem; flex-direction:column; text-align:center }
      nav{ display:none }
    }

    /* Reserve modal */
    .modal{ position:fixed; left:0; top:0; right:0; bottom:0; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.6); opacity:0; pointer-events:none; transition:opacity .2s }
    .modal.open{ opacity:1; pointer-events:auto }
    .modal .card{ background:#fff; color:#222; padding:1.6rem; border-radius:8px; min-width:280px; max-width:420px }
    .close{ background:#eee;border-radius:6px;padding:.4rem .6rem;cursor:pointer }
  </style>
</head>
<body>
  <div class="overlay">
    <header>
      <h1>Ресторан "У моря"</h1>
      <nav>
        <a href="#menu">Меню</a>
        <a href="#contacts">Контакты</a>
        <a href="#reserve" id="reserveBtn" class="btn">Забронировать</a>
      </nav>
    </header>

    <main>
      <section class="hero">
        <div class="intro">
          <h2>Свежая еда, уютная атмосфера и вид на море</h2>
          <p>Насладитесь авторскими блюдами из местных ингредиентов, приготовленными с любовью нашим шеф-поваром.</p>
          <p><strong>Часы работы:</strong> 10:00 — 23:00</p>
          <p>Текущее время: <span id="now"></span></p>
          <a href="#menu" class="btn">Посмотреть меню</a>
        </div>
        <div class="visual" aria-hidden="true">
          <img src="https://images.unsplash.com/photo-1498654896293-37aacf113fd9?w=800&q=80&auto=format&fit=crop" alt="restaurant" style="width:320px;border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,0.6)">
        </div>
      </section>

      <section id="menu" class="section">
        <h3>Меню</h3>
        <div class="menu">
          <div class="menu-item"><h4>Салат из лосося</h4><p>Свежий лосось, микс салатов, цитрусовая заправка — 320 ₴</p></div>
          <div class="menu-item"><h4>Паста морская</h4><p>Паста с морепродуктами в сливочном соусе — 380 ₴</p></div>
          <div class="menu-item"><h4>Стейк</h4><p>Говяжий стейк с овощами — 420 ₴</p></div>
          <div class="menu-item"><h4>Десерт дня</h4><p>Творожный чизкейк с ягодным соусом — 150 ₴</p></div>
        </div>
      </section>

      <section id="contacts" class="section">
        <h3>Контакты</h3>
        <p>Адрес: Одесса, набережная, 1</p>
        <p>Телефон: <a href="tel:+380000000000" style="color:var(--muted);text-decoration:underline">+38 0XX XXX XX XX</a></p>
      </section>
    </main>

    <footer>
      © Ресторан "У моря" — Приятного аппетита!
    </footer>

    <div id="reserveModal" class="modal" role="dialog" aria-hidden="true">
      <div class="card">
        <h4>Бронирование</h4>
        <p>Пожалуйста, оставьте номер — мы свяжемся с вами для подтверждения.</p>
        <div style="display:flex;gap:.5rem;margin-top:.6rem">
          <input id="phone" placeholder="Ваш телефон" style="flex:1;padding:.5rem;border:1px solid #ddd;border-radius:6px">
          <button id="sendReserve" class="btn">Отправить</button>
        </div>
        <div style="margin-top:.6rem;text-align:right"><button id="closeModal" class="close">Закрыть</button></div>
      </div>
    </div>
  </div>

  <script>
    // Live clock
    document.addEventListener('DOMContentLoaded', function(){
      var nowEl = document.getElementById('now');
      function tick(){ nowEl.textContent = new Date().toLocaleTimeString(); }
      tick(); setInterval(tick,1000);

      var reserveBtn = document.getElementById('reserveBtn');
      var modal = document.getElementById('reserveModal');
      var close = document.getElementById('closeModal');
      var send = document.getElementById('sendReserve');
      reserveBtn && reserveBtn.addEventListener('click', function(e){ e.preventDefault(); modal.classList.add('open'); modal.setAttribute('aria-hidden','false'); });
      close && close.addEventListener('click', function(){ modal.classList.remove('open'); modal.setAttribute('aria-hidden','true'); });
      send && send.addEventListener('click', function(){ alert('Спасибо! Мы свяжемся с вами для подтверждения брони.'); modal.classList.remove('open'); modal.setAttribute('aria-hidden','true'); });

      // Smooth scroll for anchor links
      document.querySelectorAll('a[href^="#"]').forEach(function(a){ a.addEventListener('click', function(e){ var target = document.querySelector(this.getAttribute('href')); if(target){ e.preventDefault(); target.scrollIntoView({behavior:'smooth'}); } }); });
    });
  </script>
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
