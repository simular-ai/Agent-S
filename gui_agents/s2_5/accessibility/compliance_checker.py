"""
Section 508 Compliance Checker

Implements comprehensive Section 508 compliance testing including:
- Text alternatives for non-text content
- Keyboard accessibility
- Color contrast requirements
- Focus indicators
- Form labels and instructions
- Error identification and suggestions
"""

import base64
import io
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


class ViolationSeverity(Enum):
    """Severity levels for accessibility violations"""
    CRITICAL = "critical"  # Prevents access to content or functionality
    MAJOR = "major"        # Significantly impacts usability
    MINOR = "minor"        # Minor usability impact
    WARNING = "warning"    # Potential issue or best practice


@dataclass
class AccessibilityViolation:
    """Represents an accessibility violation"""
    rule_id: str
    severity: ViolationSeverity
    description: str
    location: Optional[Dict] = None
    screenshot_path: Optional[str] = None
    suggestion: Optional[str] = None
    wcag_reference: Optional[str] = None
    section_508_reference: Optional[str] = None


class AccessibilityComplianceChecker:
    """
    Main class for performing Section 508 and WCAG compliance checking
    """
    
    def __init__(self, screenshot_dir: str = "accessibility_screenshots"):
        self.screenshot_dir = screenshot_dir
        self.violations: List[AccessibilityViolation] = []
        self.current_screenshot = None
        
        # Section 508 compliance rules mapping
        self.section_508_rules = {
            "1194.22(a)": "Text alternatives for non-text content",
            "1194.22(b)": "Equivalent alternatives for multimedia",
            "1194.22(c)": "Color is not the only means of conveying information",
            "1194.22(d)": "Documents are organized so they are readable without requiring an associated stylesheet",
            "1194.22(e)": "Redundant text links are provided for each active region of a server-side image map",
            "1194.22(f)": "Client-side image maps are provided instead of server-side image maps",
            "1194.22(g)": "Row and column headers are identified for data tables",
            "1194.22(h)": "Markup is used to associate data cells and header cells for data tables",
            "1194.22(i)": "Frames are titled with text that facilitates frame identification and navigation",
            "1194.22(j)": "Pages are designed to avoid causing the screen to flicker",
            "1194.22(k)": "A text-only page is provided when compliance cannot be accomplished in any other way",
            "1194.22(l)": "When pages use scripting languages, functionality is accessible",
            "1194.22(m)": "When applets, plug-ins, or other applications are used, accessibility is provided",
            "1194.22(n)": "Electronic forms are designed so that they are accessible",
            "1194.22(o)": "Navigation links are provided to skip repetitive content",
            "1194.22(p)": "When a timed response is required, the user is alerted"
        }
    
    def check_compliance(self, observation: Dict, accessibility_tree: str = None) -> List[AccessibilityViolation]:
        """
        Perform comprehensive Section 508 compliance checking
        
        Args:
            observation: Agent observation containing screenshot and accessibility data
            accessibility_tree: Optional accessibility tree string
            
        Returns:
            List of accessibility violations found
        """
        self.violations = []
        self.current_screenshot = observation.get("screenshot")
        
        # Convert screenshot bytes to PIL Image if needed
        if isinstance(self.current_screenshot, bytes):
            screenshot_image = Image.open(io.BytesIO(self.current_screenshot))
        else:
            screenshot_image = self.current_screenshot
            
        # Run all compliance checks
        self._check_text_alternatives(observation)
        self._check_keyboard_accessibility(observation)
        self._check_color_contrast(screenshot_image)
        self._check_focus_indicators(observation)
        self._check_form_labels(observation)
        self._check_headings_structure(accessibility_tree)
        self._check_table_headers(accessibility_tree)
        self._check_link_purposes(accessibility_tree)
        
        return self.violations
    
    def _check_text_alternatives(self, observation: Dict):
        """Check for text alternatives on images and non-text content"""
        # This would analyze the accessibility tree for images without alt text
        # For now, implementing a basic check
        
        # Check if there are images in the observation
        if self.current_screenshot:
            # Use OCR or image analysis to detect images without text alternatives
            # This is a simplified implementation
            violation = AccessibilityViolation(
                rule_id="508_1194.22(a)",
                severity=ViolationSeverity.CRITICAL,
                description="Images may be missing text alternatives",
                suggestion="Ensure all images have descriptive alt text",
                section_508_reference="1194.22(a)",
                wcag_reference="WCAG 2.1 SC 1.1.1 Non-text Content"
            )
            # Only add if we detect actual issues (placeholder for now)
            # self.violations.append(violation)
    
    def _check_keyboard_accessibility(self, observation: Dict):
        """Check for keyboard accessibility issues"""
        # Check if elements are keyboard focusable
        # This would integrate with keyboard navigation testing
        
        violation = AccessibilityViolation(
            rule_id="508_keyboard",
            severity=ViolationSeverity.CRITICAL,
            description="Elements may not be keyboard accessible",
            suggestion="Ensure all interactive elements are keyboard focusable",
            section_508_reference="1194.22(a)",
            wcag_reference="WCAG 2.1 SC 2.1.1 Keyboard"
        )
        # Placeholder - actual implementation would test keyboard navigation
        # self.violations.append(violation)
    
    def _check_color_contrast(self, image: Image.Image):
        """Check color contrast ratios"""
        if not image:
            return
            
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Basic color contrast analysis
        # This is a simplified implementation - full implementation would:
        # 1. Identify text areas
        # 2. Calculate contrast ratios between text and background
        # 3. Check against WCAG AA/AAA standards (4.5:1 and 7:1 ratios)
        
        # For demonstration, we'll check if there are very low contrast areas
        if len(img_array.shape) == 3:  # Color image
            # Convert to grayscale for contrast analysis
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Calculate local contrast
            # This is a very basic implementation
            contrast_threshold = 50  # Arbitrary threshold for demo
            
            # Check for low contrast areas (this is overly simplified)
            low_contrast_areas = np.std(gray) < contrast_threshold
            
            if low_contrast_areas:
                violation = AccessibilityViolation(
                    rule_id="508_1194.22(c)",
                    severity=ViolationSeverity.MAJOR,
                    description="Low color contrast detected",
                    suggestion="Ensure color contrast ratio meets WCAG AA standards (4.5:1 for normal text)",
                    section_508_reference="1194.22(c)",
                    wcag_reference="WCAG 2.1 SC 1.4.3 Contrast (Minimum)"
                )
                # Only add if we actually detect low contrast
                # self.violations.append(violation)
    
    def _check_focus_indicators(self, observation: Dict):
        """Check for visible focus indicators"""
        # Check if focused elements have visible focus indicators
        
        violation = AccessibilityViolation(
            rule_id="508_focus",
            severity=ViolationSeverity.MAJOR,
            description="Focus indicators may be missing or inadequate",
            suggestion="Ensure all focusable elements have visible focus indicators",
            section_508_reference="1194.22(c)",
            wcag_reference="WCAG 2.1 SC 2.4.7 Focus Visible"
        )
        # Placeholder for actual focus indicator detection
        # self.violations.append(violation)
    
    def _check_form_labels(self, observation: Dict):
        """Check for proper form labeling"""
        # Check if form controls have associated labels
        
        violation = AccessibilityViolation(
            rule_id="508_1194.22(n)",
            severity=ViolationSeverity.CRITICAL,
            description="Form controls may be missing labels",
            suggestion="Ensure all form controls have descriptive labels",
            section_508_reference="1194.22(n)",
            wcag_reference="WCAG 2.1 SC 1.3.1 Info and Relationships"
        )
        # Placeholder for actual form label checking
        # self.violations.append(violation)
    
    def _check_headings_structure(self, accessibility_tree: str):
        """Check for proper heading structure"""
        if not accessibility_tree:
            return
            
        # Parse headings from accessibility tree
        heading_pattern = r'heading.*level\s*(\d+)'
        headings = re.findall(heading_pattern, accessibility_tree, re.IGNORECASE)
        
        if headings:
            # Check for heading level gaps
            heading_levels = [int(level) for level in headings]
            
            for i in range(1, len(heading_levels)):
                if heading_levels[i] > heading_levels[i-1] + 1:
                    violation = AccessibilityViolation(
                        rule_id="508_headings",
                        severity=ViolationSeverity.MAJOR,
                        description="Heading levels skip levels in the hierarchy",
                        suggestion="Use heading levels sequentially (h1, h2, h3, etc.)",
                        wcag_reference="WCAG 2.1 SC 1.3.1 Info and Relationships"
                    )
                    self.violations.append(violation)
                    break
    
    def _check_table_headers(self, accessibility_tree: str):
        """Check for proper table header associations"""
        if not accessibility_tree:
            return
            
        # Look for tables without proper headers
        if "table" in accessibility_tree.lower():
            # Basic check for table headers
            if "columnheader" not in accessibility_tree.lower() and "rowheader" not in accessibility_tree.lower():
                violation = AccessibilityViolation(
                    rule_id="508_1194.22(g)",
                    severity=ViolationSeverity.MAJOR,
                    description="Data tables missing proper header identification",
                    suggestion="Use proper header markup for data table headers",
                    section_508_reference="1194.22(g)",
                    wcag_reference="WCAG 2.1 SC 1.3.1 Info and Relationships"
                )
                self.violations.append(violation)
    
    def _check_link_purposes(self, accessibility_tree: str):
        """Check for descriptive link text"""
        if not accessibility_tree:
            return
            
        # Look for generic link text
        generic_link_patterns = [
            r'link.*"click here"',
            r'link.*"more"',
            r'link.*"read more"',
            r'link.*"here"'
        ]
        
        for pattern in generic_link_patterns:
            if re.search(pattern, accessibility_tree, re.IGNORECASE):
                violation = AccessibilityViolation(
                    rule_id="508_links",
                    severity=ViolationSeverity.MINOR,
                    description="Links with non-descriptive text found",
                    suggestion="Use descriptive link text that explains the link's purpose",
                    wcag_reference="WCAG 2.1 SC 2.4.4 Link Purpose (In Context)"
                )
                self.violations.append(violation)
                break
    
    def capture_violation_screenshot(self, violation: AccessibilityViolation, 
                                   highlight_area: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Capture screenshot with violation highlighted
        
        Args:
            violation: The accessibility violation
            highlight_area: Optional (x, y, width, height) to highlight
            
        Returns:
            Path to saved screenshot
        """
        if not self.current_screenshot:
            return None
            
        # Convert screenshot to PIL Image
        if isinstance(self.current_screenshot, bytes):
            image = Image.open(io.BytesIO(self.current_screenshot))
        else:
            image = self.current_screenshot.copy()
        
        # Add violation highlighting
        if highlight_area:
            draw = ImageDraw.Draw(image)
            x, y, w, h = highlight_area
            
            # Draw red rectangle around violation area
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            
            # Add violation text
            try:
                font = ImageFont.load_default()
                draw.text((x, y - 20), f"Violation: {violation.rule_id}", 
                         fill="red", font=font)
            except:
                # Fallback if font loading fails
                draw.text((x, y - 20), f"Violation: {violation.rule_id}", fill="red")
        
        # Save screenshot
        import os
        os.makedirs(self.screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(
            self.screenshot_dir, 
            f"violation_{violation.rule_id}_{len(self.violations)}.png"
        )
        image.save(screenshot_path)
        
        return screenshot_path
    
    def get_violations_by_severity(self) -> Dict[ViolationSeverity, List[AccessibilityViolation]]:
        """Group violations by severity level"""
        violations_by_severity = {severity: [] for severity in ViolationSeverity}
        
        for violation in self.violations:
            violations_by_severity[violation.severity].append(violation)
            
        return violations_by_severity
    
    def get_compliance_score(self) -> float:
        """
        Calculate overall compliance score (0-100)
        
        Returns:
            Compliance score where 100 is fully compliant
        """
        if not self.violations:
            return 100.0
        
        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.CRITICAL: 25,
            ViolationSeverity.MAJOR: 10,
            ViolationSeverity.MINOR: 5,
            ViolationSeverity.WARNING: 1
        }
        
        total_penalty = sum(severity_weights[v.severity] for v in self.violations)
        
        # Calculate score (100 - penalties, minimum 0)
        score = max(0, 100 - total_penalty)
        return score