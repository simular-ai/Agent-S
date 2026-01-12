"""Generate Dressa-inspired site (MVP) and open index.html in default browser, then capture screenshot."""
import os, webbrowser, time
out_dir = os.path.join('results','website_dressa_inspired')
index = os.path.join(out_dir,'index.html')
if not os.path.exists(index):
    raise SystemExit('Site not found; run generator')
print('Opening', index)
webbrowser.open('file://' + os.path.abspath(index))
# wait a bit for browser
time.sleep(2)
try:
    import pyautogui
    img = pyautogui.screenshot()
    os.makedirs(out_dir, exist_ok=True)
    shot = os.path.join(out_dir,'screenshot_home.png')
    img.save(shot)
    print('Saved screenshot to', shot)
except Exception as e:
    print('Could not take screenshot:', e)
