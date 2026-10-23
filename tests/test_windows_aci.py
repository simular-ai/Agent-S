import unittest
from gui_agents.s1.aci.WindowsOSACI import WindowsACI


class TestWindowsACIHotkey(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.aci = WindowsACI(top_app_only=True, ocr=False)

    def test_hotkey_with_string_single_key(self):
        """Test that passing a string like 'enter' works correctly"""
        result = self.aci.hotkey('enter')
        expected = "import pyautogui; pyautogui.hotkey('enter', interval=0.5)"
        self.assertEqual(result, expected)

    def test_hotkey_with_list_single_key(self):
        """Test that passing a list with a single key works correctly"""
        result = self.aci.hotkey(['enter'])
        expected = "import pyautogui; pyautogui.hotkey('enter', interval=0.5)"
        self.assertEqual(result, expected)

    def test_hotkey_with_list_multiple_keys(self):
        """Test that passing a list with multiple keys works correctly"""
        result = self.aci.hotkey(['alt', 'tab'])
        expected = "import pyautogui; pyautogui.hotkey('alt', 'tab', interval=0.5)"
        self.assertEqual(result, expected)

    def test_hotkey_with_control_normalization(self):
        """Test that 'control' is normalized to 'ctrl'"""
        result = self.aci.hotkey(['control', 'c'])
        expected = "import pyautogui; pyautogui.hotkey('ctrl', 'c', interval=0.5)"
        self.assertEqual(result, expected)

    def test_hotkey_with_string_special_keys(self):
        """Test various special key strings"""
        test_cases = [
            ('escape', "import pyautogui; pyautogui.hotkey('escape', interval=0.5)"),
            ('space', "import pyautogui; pyautogui.hotkey('space', interval=0.5)"),
            ('backspace', "import pyautogui; pyautogui.hotkey('backspace', interval=0.5)"),
            ('delete', "import pyautogui; pyautogui.hotkey('delete', interval=0.5)"),
        ]
        for key, expected in test_cases:
            with self.subTest(key=key):
                result = self.aci.hotkey(key)
                self.assertEqual(result, expected)

    def test_hotkey_does_not_split_string(self):
        """Regression test: ensure 'enter' doesn't become 'e', 'n', 't', 'e', 'r'"""
        result = self.aci.hotkey('enter')
        # This should NOT contain individual characters
        self.assertNotIn("'e', 'n', 't', 'e', 'r'", result)
        # This SHOULD contain the full key name
        self.assertIn("'enter'", result)


if __name__ == "__main__":
    unittest.main()
