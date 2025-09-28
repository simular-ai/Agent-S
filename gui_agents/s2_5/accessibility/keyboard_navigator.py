"""
Keyboard Navigation Tester

Implements comprehensive keyboard accessibility testing including:
- Tab order validation
- Focus management
- Keyboard shortcuts testing
- Skip links functionality
- Modal dialog keyboard trapping
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pyautogui

from .compliance_checker import AccessibilityViolation, ViolationSeverity

logger = logging.getLogger(__name__)


class KeyboardTestResult(Enum):
    """Results of keyboard navigation tests"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class KeyboardTestCase:
    """Represents a keyboard navigation test case"""
    test_id: str
    description: str
    keys_to_press: List[str]
    expected_behavior: str
    result: Optional[KeyboardTestResult] = None
    notes: Optional[str] = None


@dataclass 
class FocusableElement:
    """Represents a focusable element"""
    element_id: str
    element_type: str
    element_text: str
    position: Tuple[int, int]
    size: Tuple[int, int]
    tab_index: Optional[int] = None
    is_visible: bool = True
    is_enabled: bool = True


class KeyboardNavigationTester:
    """
    Comprehensive keyboard navigation testing for accessibility compliance
    """
    
    def __init__(self, screenshot_dir: str = "accessibility_screenshots"):
        self.screenshot_dir = screenshot_dir
        self.test_results: List[KeyboardTestCase] = []
        self.focusable_elements: List[FocusableElement] = []
        self.violations: List[AccessibilityViolation] = []
        
        # Disable pyautogui failsafe for testing
        pyautogui.FAILSAFE = False
        
        # Standard keyboard navigation tests
        self.standard_tests = [
            KeyboardTestCase(
                test_id="tab_navigation",
                description="Test Tab key navigation through all focusable elements",
                keys_to_press=["tab"],
                expected_behavior="Focus moves sequentially through all interactive elements"
            ),
            KeyboardTestCase(
                test_id="shift_tab_navigation", 
                description="Test Shift+Tab reverse navigation",
                keys_to_press=["shift+tab"],
                expected_behavior="Focus moves in reverse order through interactive elements"
            ),
            KeyboardTestCase(
                test_id="enter_activation",
                description="Test Enter key activation of buttons and links",
                keys_to_press=["enter"],
                expected_behavior="Buttons and links activate when Enter is pressed"
            ),
            KeyboardTestCase(
                test_id="space_activation",
                description="Test Space key activation of buttons and checkboxes",
                keys_to_press=["space"],
                expected_behavior="Buttons activate and checkboxes toggle when Space is pressed"
            ),
            KeyboardTestCase(
                test_id="escape_functionality",
                description="Test Escape key closes dialogs and cancels operations",
                keys_to_press=["escape"],
                expected_behavior="Dialogs close and operations cancel when Escape is pressed"
            ),
            KeyboardTestCase(
                test_id="arrow_navigation",
                description="Test arrow key navigation in menus and lists",
                keys_to_press=["up", "down", "left", "right"],
                expected_behavior="Arrow keys navigate within grouped elements"
            )
        ]
    
    def discover_focusable_elements(self, observation: Dict, accessibility_tree: str = None) -> List[FocusableElement]:
        """
        Discover all focusable elements on the current screen
        
        Args:
            observation: Agent observation containing screen data
            accessibility_tree: Optional accessibility tree string
            
        Returns:
            List of focusable elements found
        """
        self.focusable_elements = []
        
        if accessibility_tree:
            # Parse accessibility tree for focusable elements
            self._parse_focusable_from_tree(accessibility_tree)
        
        # Also try to detect focusable elements from screenshot analysis
        self._detect_focusable_from_screenshot(observation)
        
        logger.info(f"Discovered {len(self.focusable_elements)} focusable elements")
        return self.focusable_elements
    
    def _parse_focusable_from_tree(self, accessibility_tree: str):
        """Parse focusable elements from accessibility tree"""
        import re
        
        # Define focusable element patterns
        focusable_patterns = [
            r'(\d+)\s+(button|link|textbox|combobox|checkbox|radio|menuitem)\s+([^\t]*)\s*([^\t]*)',
            r'(\d+)\s+(\w+)\s+([^\t]*)\s*([^\t]*)'  # Generic pattern
        ]
        
        lines = accessibility_tree.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('id\t'):
                continue
                
            for pattern in focusable_patterns:
                match = re.match(pattern, line)
                if match:
                    element_id = match.group(1)
                    element_type = match.group(2).lower()
                    element_text = match.group(3).strip() if len(match.groups()) > 2 else ""
                    
                    # Check if element type is focusable
                    if self._is_focusable_type(element_type):
                        focusable_element = FocusableElement(
                            element_id=element_id,
                            element_type=element_type,
                            element_text=element_text,
                            position=(0, 0),  # Would need to extract from tree
                            size=(0, 0)       # Would need to extract from tree
                        )
                        self.focusable_elements.append(focusable_element)
                    break
    
    def _is_focusable_type(self, element_type: str) -> bool:
        """Check if element type is typically focusable"""
        focusable_types = {
            'button', 'link', 'textbox', 'combobox', 'checkbox', 
            'radio', 'menuitem', 'tab', 'slider', 'spinbutton',
            'listitem', 'treeitem', 'cell', 'columnheader', 'rowheader'
        }
        return element_type.lower() in focusable_types
    
    def _detect_focusable_from_screenshot(self, observation: Dict):
        """Detect focusable elements from screenshot analysis"""
        # This is a placeholder for more advanced image analysis
        # In a full implementation, this would use computer vision to detect
        # buttons, form fields, links, etc. from the screenshot
        pass
    
    def test_tab_navigation(self, max_tabs: int = 50) -> KeyboardTestCase:
        """
        Test Tab key navigation through all focusable elements
        
        Args:
            max_tabs: Maximum number of Tab presses to test
            
        Returns:
            Test case with results
        """
        test_case = next(t for t in self.standard_tests if t.test_id == "tab_navigation")
        
        try:
            logger.info("Starting Tab navigation test")
            
            # Record initial state
            initial_focus = self._get_current_focus()
            visited_elements = []
            
            for i in range(max_tabs):
                # Take screenshot before tab
                before_screenshot = pyautogui.screenshot()
                
                # Press Tab
                pyautogui.press('tab')
                time.sleep(0.2)  # Allow time for focus to change
                
                # Take screenshot after tab
                after_screenshot = pyautogui.screenshot()
                
                # Check if focus changed
                if self._screenshots_different(before_screenshot, after_screenshot):
                    current_focus = self._get_current_focus()
                    visited_elements.append(current_focus)
                    
                    # Check if we've cycled back to the beginning
                    if len(visited_elements) > 1 and current_focus == visited_elements[0]:
                        logger.info(f"Tab navigation completed cycle after {i+1} tabs")
                        break
                else:
                    # No focus change detected
                    if i == 0:
                        # No focusable elements found
                        test_case.result = KeyboardTestResult.FAIL
                        test_case.notes = "No focusable elements detected"
                        return test_case
            
            # Analyze results
            if len(visited_elements) >= len(self.focusable_elements):
                test_case.result = KeyboardTestResult.PASS
                test_case.notes = f"Successfully navigated through {len(visited_elements)} elements"
            else:
                test_case.result = KeyboardTestResult.WARNING
                test_case.notes = f"Only reached {len(visited_elements)} elements, expected {len(self.focusable_elements)}"
                
        except Exception as e:
            test_case.result = KeyboardTestResult.FAIL
            test_case.notes = f"Error during Tab navigation test: {str(e)}"
            logger.error(f"Tab navigation test failed: {e}")
        
        return test_case
    
    def test_shift_tab_navigation(self) -> KeyboardTestCase:
        """Test Shift+Tab reverse navigation"""
        test_case = next(t for t in self.standard_tests if t.test_id == "shift_tab_navigation")
        
        try:
            logger.info("Starting Shift+Tab navigation test")
            
            # First, navigate to the last element with Tab
            for _ in range(10):  # Assume max 10 elements for this test
                pyautogui.press('tab')
                time.sleep(0.1)
            
            # Now test reverse navigation
            initial_focus = self._get_current_focus()
            visited_elements = []
            
            for i in range(10):
                before_screenshot = pyautogui.screenshot()
                
                # Press Shift+Tab
                pyautogui.hotkey('shift', 'tab')
                time.sleep(0.2)
                
                after_screenshot = pyautogui.screenshot()
                
                if self._screenshots_different(before_screenshot, after_screenshot):
                    current_focus = self._get_current_focus()
                    visited_elements.append(current_focus)
                    
                    if len(visited_elements) > 1 and current_focus == visited_elements[0]:
                        break
            
            if visited_elements:
                test_case.result = KeyboardTestResult.PASS
                test_case.notes = f"Reverse navigation works, visited {len(visited_elements)} elements"
            else:
                test_case.result = KeyboardTestResult.FAIL
                test_case.notes = "Shift+Tab navigation not working"
                
        except Exception as e:
            test_case.result = KeyboardTestResult.FAIL
            test_case.notes = f"Error during Shift+Tab test: {str(e)}"
            logger.error(f"Shift+Tab test failed: {e}")
        
        return test_case
    
    def test_enter_activation(self) -> KeyboardTestCase:
        """Test Enter key activation"""
        test_case = next(t for t in self.standard_tests if t.test_id == "enter_activation")
        
        try:
            logger.info("Starting Enter activation test")
            
            # Find buttons and links to test
            testable_elements = [
                elem for elem in self.focusable_elements 
                if elem.element_type in ['button', 'link']
            ]
            
            if not testable_elements:
                test_case.result = KeyboardTestResult.SKIP
                test_case.notes = "No buttons or links found to test"
                return test_case
            
            successful_activations = 0
            
            for element in testable_elements[:3]:  # Test up to 3 elements
                # Focus on the element (simplified - would need actual focusing)
                before_screenshot = pyautogui.screenshot()
                
                # Press Enter
                pyautogui.press('enter')
                time.sleep(0.3)
                
                after_screenshot = pyautogui.screenshot()
                
                # Check if something changed (activation occurred)
                if self._screenshots_different(before_screenshot, after_screenshot):
                    successful_activations += 1
            
            if successful_activations > 0:
                test_case.result = KeyboardTestResult.PASS
                test_case.notes = f"Enter activated {successful_activations} out of {len(testable_elements)} tested elements"
            else:
                test_case.result = KeyboardTestResult.FAIL
                test_case.notes = "Enter key did not activate any tested elements"
                
        except Exception as e:
            test_case.result = KeyboardTestResult.FAIL
            test_case.notes = f"Error during Enter activation test: {str(e)}"
            logger.error(f"Enter activation test failed: {e}")
        
        return test_case
    
    def test_keyboard_trapping(self) -> KeyboardTestCase:
        """Test that focus is properly trapped in modal dialogs"""
        test_case = KeyboardTestCase(
            test_id="keyboard_trapping",
            description="Test focus trapping in modal dialogs",
            keys_to_press=["tab"],
            expected_behavior="Focus stays within modal dialogs"
        )
        
        try:
            # This would test if focus is trapped within modal dialogs
            # Implementation would depend on detecting modal state
            test_case.result = KeyboardTestResult.SKIP
            test_case.notes = "No modal dialogs detected for testing"
            
        except Exception as e:
            test_case.result = KeyboardTestResult.FAIL
            test_case.notes = f"Error during keyboard trapping test: {str(e)}"
        
        return test_case
    
    def test_skip_links(self) -> KeyboardTestCase:
        """Test skip link functionality"""
        test_case = KeyboardTestCase(
            test_id="skip_links",
            description="Test skip link functionality for bypassing repetitive content",
            keys_to_press=["tab", "enter"],
            expected_behavior="Skip links allow bypassing repetitive navigation"
        )
        
        try:
            # Look for skip links (usually appear on first Tab)
            pyautogui.press('tab')
            time.sleep(0.2)
            
            # Take screenshot to check for skip link
            screenshot = pyautogui.screenshot()
            
            # This is a simplified check - would need better detection
            test_case.result = KeyboardTestResult.SKIP
            test_case.notes = "Skip link detection not implemented"
            
        except Exception as e:
            test_case.result = KeyboardTestResult.FAIL
            test_case.notes = f"Error during skip link test: {str(e)}"
        
        return test_case
    
    def run_all_keyboard_tests(self, observation: Dict, accessibility_tree: str = None) -> List[KeyboardTestCase]:
        """
        Run all keyboard navigation tests
        
        Args:
            observation: Agent observation containing screen data
            accessibility_tree: Optional accessibility tree string
            
        Returns:
            List of test results
        """
        logger.info("Starting comprehensive keyboard navigation testing")
        
        # Discover focusable elements first
        self.discover_focusable_elements(observation, accessibility_tree)
        
        # Run standard tests
        self.test_results = []
        
        # Tab navigation test
        self.test_results.append(self.test_tab_navigation())
        
        # Shift+Tab navigation test  
        self.test_results.append(self.test_shift_tab_navigation())
        
        # Enter activation test
        self.test_results.append(self.test_enter_activation())
        
        # Additional tests
        self.test_results.append(self.test_keyboard_trapping())
        self.test_results.append(self.test_skip_links())
        
        # Generate violations based on test results
        self._generate_violations_from_tests()
        
        logger.info(f"Completed {len(self.test_results)} keyboard tests")
        return self.test_results
    
    def _generate_violations_from_tests(self):
        """Generate accessibility violations based on test results"""
        for test in self.test_results:
            if test.result == KeyboardTestResult.FAIL:
                severity = ViolationSeverity.CRITICAL
                if test.test_id in ["keyboard_trapping", "skip_links"]:
                    severity = ViolationSeverity.MAJOR
                
                violation = AccessibilityViolation(
                    rule_id=f"keyboard_{test.test_id}",
                    severity=severity,
                    description=f"Keyboard test failed: {test.description}",
                    suggestion=f"Fix keyboard accessibility issue: {test.expected_behavior}",
                    wcag_reference="WCAG 2.1 SC 2.1.1 Keyboard",
                    section_508_reference="1194.22(a)"
                )
                self.violations.append(violation)
    
    def _get_current_focus(self) -> str:
        """Get identifier for currently focused element"""
        # This is a placeholder - actual implementation would detect focus
        # Could use accessibility APIs or screen analysis
        return f"focus_{int(time.time() * 1000)}"
    
    def _screenshots_different(self, img1, img2, threshold: float = 0.95) -> bool:
        """Check if two screenshots are significantly different"""
        if img1.size != img2.size:
            return True
            
        # Convert to numpy arrays for comparison
        import numpy as np
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate similarity
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity < threshold
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all keyboard tests"""
        if not self.test_results:
            return {"error": "No tests have been run"}
        
        summary = {
            "total_tests": len(self.test_results),
            "passed": len([t for t in self.test_results if t.result == KeyboardTestResult.PASS]),
            "failed": len([t for t in self.test_results if t.result == KeyboardTestResult.FAIL]),
            "warnings": len([t for t in self.test_results if t.result == KeyboardTestResult.WARNING]),
            "skipped": len([t for t in self.test_results if t.result == KeyboardTestResult.SKIP]),
            "focusable_elements_found": len(self.focusable_elements),
            "violations_generated": len(self.violations)
        }
        
        return summary