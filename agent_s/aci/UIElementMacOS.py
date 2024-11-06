from Foundation import *
from AppKit import *
import os

from ApplicationServices import (
    AXIsProcessTrusted,
    AXUIElementCreateApplication,
    AXUIElementCreateSystemWide,
    CFEqual,
)

from ApplicationServices import (
    AXUIElementCopyAttributeNames,
    AXUIElementCopyAttributeValue,
)

from AppKit import NSWorkspace, NSRunningApplication

import logging

logger = logging.getLogger("openaci.agent")


class UIElement(object):

    def __init__(self, ref=None):
        self.ref = ref

    def getAttributeNames(self):
        error_code, attributeNames = AXUIElementCopyAttributeNames(self.ref, None)
        return list(attributeNames)

    def attribute(self, key: str):
        error, value = AXUIElementCopyAttributeValue(self.ref, key, None)
        return value

    def children(self):
        return self.attribute("AXChildren")

    def systemWideElement():
        ref = AXUIElementCreateSystemWide()
        return UIElement(ref)

    def role(self):
        return self.attribute("AXRole")

    def position(self):
        pos = self.attribute("AXPosition")
        pos_parts = pos.__repr__().split().copy()
        # Find the parts containing 'x:' and 'y:'
        x_part = next(part for part in pos_parts if part.startswith("x:"))
        y_part = next(part for part in pos_parts if part.startswith("y:"))

        # Extract the numerical values after 'x:' and 'y:'
        x = float(x_part.split(":")[1])
        y = float(y_part.split(":")[1])

        return (x, y)

    def size(self):
        size = self.attribute("AXSize")
        size_parts = size.__repr__().split().copy()
        # Find the parts containing 'Width:' and 'Height:'
        width_part = next(part for part in size_parts if part.startswith("w:"))
        height_part = next(part for part in size_parts if part.startswith("h:"))

        # Extract the numerical values after 'Width:' and 'Height:'
        w = float(width_part.split(":")[1])
        h = float(height_part.split(":")[1])
        return (w, h)
    
    def isValid(self):
        if self.position() is not None and self.size() is not None:
            return True

    def parse(self, element):
        position = element.position(element)
        size = element.size(element)
        return {
            "position": position,
            "size": size,
            "title": str(element.attribute("AXTitle")),
            "text": str(element.attribute("AXDescription"))
            or str(element.attribute("AXValue")),
            "role": str(element.attribute("AXRole")),
        }

    @staticmethod
    def get_current_applications(obs: Dict):
        # Get the shared workspace instance
        workspace = NSWorkspace.sharedWorkspace()

        # Get a list of running applications
        running_apps = workspace.runningApplications()

        # Iterate through the list and print each application's name
        current_apps = []
        for app in running_apps:
            if app.activationPolicy() == 0:
                app_name = app.localizedName()
                current_apps.append(app_name)

        return current_apps

    @staticmethod
    def list_apps_in_directories():
        directories_to_search = ["/System/Applications", "/Applications"]
        apps = []
        for directory in directories_to_search:
            if os.path.exists(directory):
                directory_apps = [
                    app for app in os.listdir(directory) if app.endswith(".app")
                ]
                apps.extend(directory_apps)
        return apps

    @staticmethod
    def get_top_app():
        return NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()

    def __repr__(self):
        return "UIElement%s" % (self.ref)


if __name__ == "__main__":
    # Examples.
    elem = UIElement.systemWideElement()
    print(elem)
    print(elem.attribute("AXFocusedApplication"))
    print(elem.getAttributeNames())
    elem = UIElement(elem.attribute("AXFocusedApplication"))
    print(elem.getAttributeNames())
