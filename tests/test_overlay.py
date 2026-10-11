import os
import unittest
from unittest.mock import patch
from gui_agents.s3.overlay import OverlayController


class OverlayTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows Tk lifecycle smoke test")
    def test_withdrawn_process_lifecycle(self):
        for _ in range(3):
            controller = OverlayController()
            # Initialization starts withdrawn; suppress the initial show.
            with patch.object(controller, "restore"):
                status = controller.show(
                    "synthetic",
                    "stub",
                    {"left": 0, "top": 0, "width": 1920, "height": 1080},
                )
            self.assertEqual(status, "ok")
            process = controller.process
            controller.update("test", "synthetic action")
            controller.withdraw()
            controller.close()
            self.assertEqual(process.returncode, 0)
