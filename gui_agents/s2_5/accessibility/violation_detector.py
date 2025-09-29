"""
Violation Detection System

Implements real-time accessibility violation detection during agent operations including:
- Screenshot-based violation detection
- DOM analysis for accessibility issues
- Focus state monitoring
- Color contrast analysis
- Interactive element validation
"""

import base64
import io
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageEnhance
import cv2

from .compliance_checker import AccessibilityViolation, ViolationSeverity

logger = logging.getLogger(__name__)


@dataclass
class ViolationEvidence:
    """Evidence captured for an accessibility violation"""
    violation_id: str
    timestamp: float
    screenshot_path: str
    description: str
    coordinates: Optional[Tuple[int, int, int, int]] = None
    additional_data: Optional[Dict] = None


class ViolationDetector:
    """
    Real-time accessibility violation detection system
    """
    
    def __init__(self, screenshot_dir: str = "accessibility_screenshots"):
        self.screenshot_dir = screenshot_dir
        self.violations: List[AccessibilityViolation] = []
        self.evidence: List[ViolationEvidence] = []
        self.previous_screenshot = None
        self.current_focus_element = None
        
        # Violation detection settings
        self.detection_enabled = True
        self.capture_evidence = True
        self.real_time_monitoring = True
        
        # Color contrast thresholds (WCAG standards)
        self.contrast_ratios = {
            'aa_normal': 4.5,      # WCAG AA for normal text
            'aa_large': 3.0,       # WCAG AA for large text  
            'aaa_normal': 7.0,     # WCAG AAA for normal text
            'aaa_large': 4.5       # WCAG AAA for large text
        }
    
    def analyze_screenshot_for_violations(self, screenshot: bytes, 
                                        accessibility_tree: str = None) -> List[AccessibilityViolation]:
        """
        Analyze screenshot for accessibility violations
        
        Args:
            screenshot: Screenshot bytes
            accessibility_tree: Optional accessibility tree data
            
        Returns:
            List of violations detected
        """
        if not self.detection_enabled:
            return []
        
        violations = []
        
        # Convert screenshot to PIL Image
        if isinstance(screenshot, bytes):
            image = Image.open(io.BytesIO(screenshot))
        else:
            image = screenshot
            
        # Run various violation detection methods
        violations.extend(self._detect_color_contrast_violations(image))
        violations.extend(self._detect_missing_focus_indicators(image))
        violations.extend(self._detect_small_click_targets(image))
        violations.extend(self._detect_text_alternatives_violations(image, accessibility_tree))
        violations.extend(self._detect_keyboard_traps(image))
        
        # Capture evidence for each violation
        if self.capture_evidence:
            for violation in violations:
                self._capture_violation_evidence(violation, image)
        
        self.violations.extend(violations)
        self.previous_screenshot = image
        
        return violations
    
    def _detect_color_contrast_violations(self, image: Image.Image) -> List[AccessibilityViolation]:
        """Detect color contrast violations using image analysis"""
        violations = []
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to LAB color space for better contrast analysis
            if len(img_array.shape) == 3:
                # Use OpenCV for more accurate color space conversion
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                
                # Analyze luminance channel (L)
                luminance = img_lab[:, :, 0]
                
                # Detect text-like regions using edge detection
                edges = cv2.Canny(luminance, 50, 150)
                
                # Find contours that might be text
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filter contours by size (potential text regions)
                    area = cv2.contourArea(contour)
                    if 50 < area < 5000:  # Reasonable text size range
                        
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Extract text region and background
                        text_region = luminance[y:y+h, x:x+w]
                        
                        # Calculate contrast ratio
                        if text_region.size > 0:
                            text_luminance = np.mean(text_region)
                            
                            # Estimate background by expanding region
                            bg_y1 = max(0, y - 10)
                            bg_y2 = min(luminance.shape[0], y + h + 10)
                            bg_x1 = max(0, x - 10)
                            bg_x2 = min(luminance.shape[1], x + w + 10)
                            
                            bg_region = luminance[bg_y1:bg_y2, bg_x1:bg_x2]
                            bg_luminance = np.mean(bg_region)
                            
                            # Calculate contrast ratio
                            contrast_ratio = self._calculate_contrast_ratio(text_luminance, bg_luminance)
                            
                            # Check against WCAG standards
                            if contrast_ratio < self.contrast_ratios['aa_normal']:
                                severity = ViolationSeverity.MAJOR
                                if contrast_ratio < 3.0:
                                    severity = ViolationSeverity.CRITICAL
                                
                                violation = AccessibilityViolation(
                                    rule_id="color_contrast_violation",
                                    severity=severity,
                                    description=f"Low color contrast detected (ratio: {contrast_ratio:.2f})",
                                    location={"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                                    suggestion=f"Increase contrast ratio to at least {self.contrast_ratios['aa_normal']}:1",
                                    wcag_reference="WCAG 2.1 SC 1.4.3 Contrast (Minimum)",
                                    section_508_reference="1194.22(c)"
                                )
                                violations.append(violation)
                                
        except Exception as e:
            logger.error(f"Error in color contrast detection: {e}")
        
        return violations
    
    def _calculate_contrast_ratio(self, luminance1: float, luminance2: float) -> float:
        """Calculate WCAG contrast ratio between two luminance values"""
        # Normalize luminance values (0-255 to 0-1)
        l1 = luminance1 / 255.0
        l2 = luminance2 / 255.0
        
        # Apply gamma correction
        l1 = self._gamma_correct(l1)
        l2 = self._gamma_correct(l2)
        
        # Calculate contrast ratio
        lighter = max(l1, l2)
        darker = min(l1, l2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def _gamma_correct(self, value: float) -> float:
        """Apply gamma correction for luminance calculation"""
        if value <= 0.03928:
            return value / 12.92
        else:
            return ((value + 0.055) / 1.055) ** 2.4
    
    def _detect_missing_focus_indicators(self, image: Image.Image) -> List[AccessibilityViolation]:
        """Detect elements that may be missing focus indicators"""
        violations = []
        
        try:
            # Convert to grayscale for edge detection
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Detect potential interactive elements using edge detection
            edges = cv2.Canny(img_array, 50, 150)
            
            # Find contours that could be buttons or interactive elements
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            button_like_contours = []
            for contour in contours:
                # Filter by area and aspect ratio to find button-like shapes
                area = cv2.contourArea(contour)
                if 500 < area < 10000:  # Reasonable button size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Button-like aspect ratios
                    if 0.3 < aspect_ratio < 5.0:
                        button_like_contours.append((x, y, w, h))
            
            # For each potential button, check if it has visible focus indicator
            for x, y, w, h in button_like_contours:
                # Extract button region
                button_region = img_array[y:y+h, x:x+w]
                
                # Check for focus indicator patterns (borders, shadows, etc.)
                has_focus_indicator = self._has_focus_indicator(button_region)
                
                if not has_focus_indicator:
                    violation = AccessibilityViolation(
                        rule_id="missing_focus_indicator",
                        severity=ViolationSeverity.MAJOR,
                        description="Interactive element may be missing focus indicator",
                        location={"x": x, "y": y, "width": w, "height": h},
                        suggestion="Ensure all interactive elements have visible focus indicators",
                        wcag_reference="WCAG 2.1 SC 2.4.7 Focus Visible",
                        section_508_reference="1194.22(c)"
                    )
                    violations.append(violation)
                    
        except Exception as e:
            logger.error(f"Error in focus indicator detection: {e}")
        
        return violations
    
    def _has_focus_indicator(self, region: np.ndarray) -> bool:
        """Check if a region has visible focus indicator patterns"""
        # This is a simplified check - would need more sophisticated analysis
        
        # Check for border patterns
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # High edge density might indicate borders/focus indicators
        return edge_density > 0.1
    
    def _detect_small_click_targets(self, image: Image.Image) -> List[AccessibilityViolation]:
        """Detect click targets that are too small (below 44x44px WCAG guideline)"""
        violations = []
        
        try:
            # Convert to grayscale
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Detect potential clickable elements
            edges = cv2.Canny(img_array, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_size = 44  # WCAG recommendation for touch targets
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if this looks like a clickable element
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # Reasonable interactive element size range
                    
                    # Check if dimensions are too small
                    if w < min_size or h < min_size:
                        violation = AccessibilityViolation(
                            rule_id="small_click_target",
                            severity=ViolationSeverity.MINOR,
                            description=f"Click target too small ({w}x{h}px, should be at least {min_size}x{min_size}px)",
                            location={"x": x, "y": y, "width": w, "height": h},
                            suggestion=f"Increase click target size to at least {min_size}x{min_size}px",
                            wcag_reference="WCAG 2.1 SC 2.5.5 Target Size"
                        )
                        violations.append(violation)
                        
        except Exception as e:
            logger.error(f"Error in click target size detection: {e}")
        
        return violations
    
    def _detect_text_alternatives_violations(self, image: Image.Image, 
                                           accessibility_tree: str = None) -> List[AccessibilityViolation]:
        """Detect images that may be missing text alternatives"""
        violations = []
        
        try:
            # Use basic image analysis to detect images
            img_array = np.array(image)
            
            # Detect image-like regions using color variance
            if len(img_array.shape) == 3:
                # Convert to HSV for better image detection
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                
                # Look for regions with high color variance (likely images)
                saturation = hsv[:, :, 1]
                
                # Find regions with high saturation variance
                kernel = np.ones((20, 20), np.uint8)
                sat_std = cv2.filter2D(saturation.astype(np.float32), -1, kernel)
                
                # Threshold to find image-like regions
                image_regions = sat_std > np.percentile(sat_std, 90)
                
                # Find contours of image regions
                contours, _ = cv2.findContours(
                    image_regions.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Significant image size
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check if this image has alt text in accessibility tree
                        has_alt_text = self._check_alt_text_in_tree(
                            accessibility_tree, x, y, w, h
                        )
                        
                        if not has_alt_text:
                            violation = AccessibilityViolation(
                                rule_id="missing_alt_text",
                                severity=ViolationSeverity.CRITICAL,
                                description="Image detected without text alternative",
                                location={"x": x, "y": y, "width": w, "height": h},
                                suggestion="Add descriptive alt text for images",
                                wcag_reference="WCAG 2.1 SC 1.1.1 Non-text Content",
                                section_508_reference="1194.22(a)"
                            )
                            violations.append(violation)
                            
        except Exception as e:
            logger.error(f"Error in text alternatives detection: {e}")
        
        return violations
    
    def _check_alt_text_in_tree(self, accessibility_tree: str, 
                               x: int, y: int, w: int, h: int) -> bool:
        """Check if accessibility tree contains alt text for image at location"""
        if not accessibility_tree:
            return False
            
        # This is a simplified check - would need better correlation
        # between screen coordinates and accessibility tree elements
        
        # Look for image elements with alt text
        import re
        image_patterns = [
            r'image.*alt[^=]*=\s*["\']([^"\']+)["\']',
            r'graphic.*name[^=]*=\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in image_patterns:
            matches = re.findall(pattern, accessibility_tree, re.IGNORECASE)
            if matches:
                # Found at least one image with alt text
                return True
        
        return False
    
    def _detect_keyboard_traps(self, image: Image.Image) -> List[AccessibilityViolation]:
        """Detect potential keyboard traps in modal dialogs"""
        violations = []
        
        try:
            # Look for modal dialog patterns
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Detect potential modal overlays (darker background areas)
            # This is a simplified detection method
            
            # Calculate image brightness histogram
            hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
            
            # Check for bimodal distribution (dark background + bright modal)
            peaks = self._find_histogram_peaks(hist)
            
            if len(peaks) >= 2:
                # Potential modal detected
                violation = AccessibilityViolation(
                    rule_id="potential_keyboard_trap",
                    severity=ViolationSeverity.WARNING,
                    description="Potential modal dialog detected - verify keyboard focus trapping",
                    suggestion="Ensure focus is properly trapped within modal dialogs",
                    wcag_reference="WCAG 2.1 SC 2.1.2 No Keyboard Trap"
                )
                violations.append(violation)
                
        except Exception as e:
            logger.error(f"Error in keyboard trap detection: {e}")
        
        return violations
    
    def _find_histogram_peaks(self, hist: np.ndarray, min_distance: int = 50) -> List[int]:
        """Find peaks in histogram"""
        peaks = []
        
        for i in range(min_distance, len(hist) - min_distance):
            if (hist[i] > hist[i - min_distance] and 
                hist[i] > hist[i + min_distance] and
                hist[i] > np.mean(hist) * 1.5):
                peaks.append(i)
        
        return peaks
    
    def _capture_violation_evidence(self, violation: AccessibilityViolation, image: Image.Image):
        """Capture screenshot evidence for a violation"""
        try:
            # Create highlighted version of image
            evidence_image = image.copy()
            draw = ImageDraw.Draw(evidence_image)
            
            # Highlight violation area if location is provided
            if violation.location:
                x = violation.location["x"]
                y = violation.location["y"] 
                w = violation.location["width"]
                h = violation.location["height"]
                
                # Draw red rectangle around violation
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                
                # Add violation label
                label_text = f"Violation: {violation.rule_id}"
                try:
                    from PIL import ImageFont
                    font = ImageFont.load_default()
                    draw.text((x, y - 20), label_text, fill="red", font=font)
                except:
                    draw.text((x, y - 20), label_text, fill="red")
            
            # Save evidence screenshot
            import os
            os.makedirs(self.screenshot_dir, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            screenshot_path = os.path.join(
                self.screenshot_dir,
                f"violation_{violation.rule_id}_{timestamp}.png"
            )
            evidence_image.save(screenshot_path)
            
            # Create evidence record
            evidence = ViolationEvidence(
                violation_id=violation.rule_id,
                timestamp=timestamp,
                screenshot_path=screenshot_path,
                description=violation.description,
                coordinates=(
                    violation.location["x"], violation.location["y"],
                    violation.location["width"], violation.location["height"]
                ) if violation.location else None
            )
            
            self.evidence.append(evidence)
            violation.screenshot_path = screenshot_path
            
        except Exception as e:
            logger.error(f"Error capturing violation evidence: {e}")
    
    def monitor_focus_changes(self, previous_obs: Dict, current_obs: Dict) -> List[AccessibilityViolation]:
        """Monitor focus changes between observations"""
        violations = []
        
        if not self.real_time_monitoring:
            return violations
        
        try:
            # Compare screenshots to detect focus changes
            if (previous_obs.get("screenshot") and current_obs.get("screenshot")):
                
                prev_img = Image.open(io.BytesIO(previous_obs["screenshot"]))
                curr_img = Image.open(io.BytesIO(current_obs["screenshot"]))
                
                # Detect focus change
                focus_changed = self._detect_focus_change(prev_img, curr_img)
                
                if focus_changed:
                    # Analyze new focus state
                    focus_violations = self._analyze_focus_state(curr_img)
                    violations.extend(focus_violations)
                    
        except Exception as e:
            logger.error(f"Error monitoring focus changes: {e}")
        
        return violations
    
    def _detect_focus_change(self, prev_img: Image.Image, curr_img: Image.Image) -> bool:
        """Detect if focus has changed between two images"""
        try:
            # Convert to grayscale
            prev_gray = np.array(prev_img.convert('L'))
            curr_gray = np.array(curr_img.convert('L'))
            
            # Calculate difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Focus changes typically create small but noticeable differences
            focus_threshold = np.mean(diff) > 5  # Adjust threshold as needed
            
            return focus_threshold
            
        except Exception as e:
            logger.error(f"Error detecting focus change: {e}")
            return False
    
    def _analyze_focus_state(self, image: Image.Image) -> List[AccessibilityViolation]:
        """Analyze current focus state for violations"""
        violations = []
        
        # Check if current focus has visible indicator
        focus_violations = self._detect_missing_focus_indicators(image)
        violations.extend(focus_violations)
        
        return violations
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all detected violations"""
        if not self.violations:
            return {"total_violations": 0}
        
        # Group by severity
        by_severity = {}
        for severity in ViolationSeverity:
            by_severity[severity.value] = len([
                v for v in self.violations if v.severity == severity
            ])
        
        # Group by rule type
        by_rule = {}
        for violation in self.violations:
            rule_type = violation.rule_id.split('_')[0]
            by_rule[rule_type] = by_rule.get(rule_type, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "by_severity": by_severity,
            "by_rule_type": by_rule,
            "evidence_captured": len(self.evidence),
            "screenshot_dir": self.screenshot_dir
        }
    
    def clear_violations(self):
        """Clear all recorded violations and evidence"""
        self.violations.clear()
        self.evidence.clear()
        logger.info("Cleared all violations and evidence")
    
    def enable_detection(self, enabled: bool = True):
        """Enable or disable violation detection"""
        self.detection_enabled = enabled
        logger.info(f"Violation detection {'enabled' if enabled else 'disabled'}")
    
    def enable_evidence_capture(self, enabled: bool = True):
        """Enable or disable evidence capture"""
        self.capture_evidence = enabled
        logger.info(f"Evidence capture {'enabled' if enabled else 'disabled'}")