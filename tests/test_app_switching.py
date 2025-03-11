import time

import pyautogui
from AppKit import NSWorkspace

from gui_agents.s1.aci.MacOSACI import MacOSACI

agent = MacOSACI()


def test_app_switching():
    app_or_file_name = "Safari"

    exec(agent.switch_applications(app_or_file_name))

    # Checking the frontmost application
    frontmost_app = NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()
    print(frontmost_app)

    # Assert to confirm Safari is the frontmost application
    assert frontmost_app == "Safari", f"Expected Safari, but got {frontmost_app}"


# Run the test
test_app_switching()
