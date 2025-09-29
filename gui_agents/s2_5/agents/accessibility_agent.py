"""
Accessibility Agent for Agent S

Integrates accessibility and 508 compliance testing into the Agent S computer use framework.
Provides real-time accessibility monitoring, testing, and reporting capabilities.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..accessibility.compliance_checker import AccessibilityComplianceChecker, AccessibilityViolation
from ..accessibility.keyboard_navigator import KeyboardNavigationTester, KeyboardTestCase
from ..accessibility.violation_detector import ViolationDetector, ViolationEvidence
from ..accessibility.report_generator import AccessibilityReportGenerator, AccessibilityTestSession

logger = logging.getLogger(__name__)


@dataclass
class AccessibilityConfig:
    """Configuration for accessibility testing"""
    enable_real_time_monitoring: bool = True
    enable_keyboard_testing: bool = True
    enable_compliance_checking: bool = True
    enable_violation_detection: bool = True
    capture_screenshots: bool = True
    screenshot_dir: str = "accessibility_screenshots"
    report_dir: str = "accessibility_reports"
    auto_generate_reports: bool = True
    compliance_level: str = "AA"  # AA or AAA


class AccessibilityAgent:
    """
    Main accessibility agent that integrates with Agent S computer use framework
    """
    
    def __init__(self, config: AccessibilityConfig = None):
        self.config = config or AccessibilityConfig()
        
        # Initialize accessibility components
        self.compliance_checker = AccessibilityComplianceChecker(
            screenshot_dir=self.config.screenshot_dir
        )
        self.keyboard_tester = KeyboardNavigationTester(
            screenshot_dir=self.config.screenshot_dir
        )
        self.violation_detector = ViolationDetector(
            screenshot_dir=self.config.screenshot_dir
        )
        self.report_generator = AccessibilityReportGenerator(
            output_dir=self.config.report_dir
        )
        
        # Test session management
        self.current_session: Optional[AccessibilityTestSession] = None
        self.session_history: List[AccessibilityTestSession] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.previous_observation = None
        
        logger.info("Accessibility Agent initialized")
    
    def start_accessibility_session(self, 
                                  app_or_url: str, 
                                  platform: str = "unknown") -> str:
        """
        Start a new accessibility testing session
        
        Args:
            app_or_url: Application name or URL being tested
            platform: Platform (windows, mac, linux)
            
        Returns:
            Session ID
        """
        session_id = f"accessibility_session_{int(time.time())}"
        
        self.current_session = AccessibilityTestSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            url_or_app=app_or_url,
            platform=platform,
            violations=[],
            keyboard_tests=[],
            evidence=[],
            compliance_score=0.0,
            test_configuration=self._get_test_configuration()
        )
        
        # Enable monitoring if configured
        if self.config.enable_real_time_monitoring:
            self.start_monitoring()
        
        logger.info(f"Started accessibility session: {session_id}")
        return session_id
    
    def end_accessibility_session(self) -> Optional[AccessibilityTestSession]:
        """
        End current accessibility testing session and generate reports
        
        Returns:
            Completed test session or None if no active session
        """
        if not self.current_session:
            logger.warning("No active accessibility session to end")
            return None
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Finalize session
        self.current_session.end_time = datetime.now()
        self.current_session.compliance_score = self.compliance_checker.get_compliance_score()
        
        # Generate reports if configured
        if self.config.auto_generate_reports:
            try:
                report_paths = self.report_generator.generate_comprehensive_report(
                    self.current_session
                )
                logger.info(f"Generated accessibility reports: {list(report_paths.keys())}")
            except Exception as e:
                logger.error(f"Error generating accessibility reports: {e}")
        
        # Store in history
        completed_session = self.current_session
        self.session_history.append(completed_session)
        self.current_session = None
        
        logger.info(f"Ended accessibility session: {completed_session.session_id}")
        return completed_session
    
    def analyze_observation(self, observation: Dict, 
                          accessibility_tree: str = None) -> Dict[str, Any]:
        """
        Analyze an observation for accessibility issues
        
        Args:
            observation: Agent observation containing screenshot and other data
            accessibility_tree: Optional accessibility tree string
            
        Returns:
            Analysis results including violations and recommendations
        """
        if not self.current_session:
            logger.warning("No active accessibility session - starting default session")
            self.start_accessibility_session("Unknown Application")
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "violations": [],
            "keyboard_issues": [],
            "compliance_score": 0.0,
            "recommendations": [],
            "evidence_captured": []
        }
        
        try:
            # Run compliance checking
            if self.config.enable_compliance_checking:
                violations = self.compliance_checker.check_compliance(
                    observation, accessibility_tree
                )
                analysis_results["violations"] = [
                    self._violation_to_dict(v) for v in violations
                ]
                self.current_session.violations.extend(violations)
            
            # Run violation detection
            if self.config.enable_violation_detection:
                detected_violations = self.violation_detector.analyze_screenshot_for_violations(
                    observation.get("screenshot"), accessibility_tree
                )
                analysis_results["violations"].extend([
                    self._violation_to_dict(v) for v in detected_violations
                ])
                self.current_session.violations.extend(detected_violations)
            
            # Monitor focus changes if we have previous observation
            if self.previous_observation and self.config.enable_real_time_monitoring:
                focus_violations = self.violation_detector.monitor_focus_changes(
                    self.previous_observation, observation
                )
                analysis_results["violations"].extend([
                    self._violation_to_dict(v) for v in focus_violations
                ])
                self.current_session.violations.extend(focus_violations)
            
            # Update compliance score
            analysis_results["compliance_score"] = self.compliance_checker.get_compliance_score()
            
            # Generate recommendations
            analysis_results["recommendations"] = self._generate_recommendations(
                self.current_session.violations
            )
            
            # Store previous observation for next analysis
            self.previous_observation = observation
            
        except Exception as e:
            logger.error(f"Error during accessibility analysis: {e}")
            analysis_results["error"] = str(e)
        
        return analysis_results
    
    def run_keyboard_accessibility_tests(self, observation: Dict, 
                                       accessibility_tree: str = None) -> Dict[str, Any]:
        """
        Run comprehensive keyboard accessibility tests
        
        Args:
            observation: Current observation
            accessibility_tree: Optional accessibility tree
            
        Returns:
            Keyboard test results
        """
        if not self.current_session:
            logger.warning("No active accessibility session for keyboard testing")
            return {"error": "No active session"}
        
        if not self.config.enable_keyboard_testing:
            return {"message": "Keyboard testing disabled"}
        
        try:
            logger.info("Running keyboard accessibility tests")
            
            # Run all keyboard tests
            test_results = self.keyboard_tester.run_all_keyboard_tests(
                observation, accessibility_tree
            )
            
            # Store results in current session
            self.current_session.keyboard_tests.extend(test_results)
            
            # Get test summary
            test_summary = self.keyboard_tester.get_test_summary()
            
            # Add any violations found during keyboard testing
            keyboard_violations = self.keyboard_tester.violations
            self.current_session.violations.extend(keyboard_violations)
            
            return {
                "test_results": [self._keyboard_test_to_dict(t) for t in test_results],
                "summary": test_summary,
                "violations_found": len(keyboard_violations)
            }
            
        except Exception as e:
            logger.error(f"Error during keyboard accessibility testing: {e}")
            return {"error": str(e)}
    
    def capture_violation_evidence(self, violation_id: str, 
                                 highlight_area: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Capture screenshot evidence for a specific violation
        
        Args:
            violation_id: ID of the violation
            highlight_area: Optional area to highlight (x, y, width, height)
            
        Returns:
            Path to captured screenshot or None if failed
        """
        if not self.config.capture_screenshots:
            return None
        
        try:
            # Find the violation
            violation = None
            for v in self.current_session.violations if self.current_session else []:
                if v.rule_id == violation_id:
                    violation = v
                    break
            
            if not violation:
                logger.warning(f"Violation {violation_id} not found")
                return None
            
            # Capture evidence
            screenshot_path = self.compliance_checker.capture_violation_screenshot(
                violation, highlight_area
            )
            
            if screenshot_path and self.current_session:
                # Create evidence record
                evidence = ViolationEvidence(
                    violation_id=violation_id,
                    timestamp=time.time(),
                    screenshot_path=screenshot_path,
                    description=violation.description,
                    coordinates=highlight_area
                )
                self.current_session.evidence.append(evidence)
            
            return screenshot_path
            
        except Exception as e:
            logger.error(f"Error capturing violation evidence: {e}")
            return None
    
    def get_accessibility_status(self) -> Dict[str, Any]:
        """
        Get current accessibility testing status
        
        Returns:
            Status information including session details and statistics
        """
        status = {
            "monitoring_active": self.monitoring_active,
            "current_session": None,
            "session_history_count": len(self.session_history),
            "configuration": {
                "real_time_monitoring": self.config.enable_real_time_monitoring,
                "keyboard_testing": self.config.enable_keyboard_testing,
                "compliance_checking": self.config.enable_compliance_checking,
                "violation_detection": self.config.enable_violation_detection,
                "screenshot_capture": self.config.capture_screenshots,
                "compliance_level": self.config.compliance_level
            }
        }
        
        if self.current_session:
            status["current_session"] = {
                "session_id": self.current_session.session_id,
                "app_or_url": self.current_session.url_or_app,
                "platform": self.current_session.platform,
                "start_time": self.current_session.start_time.isoformat(),
                "violations_count": len(self.current_session.violations),
                "keyboard_tests_count": len(self.current_session.keyboard_tests),
                "evidence_count": len(self.current_session.evidence),
                "compliance_score": self.current_session.compliance_score
            }
        
        return status
    
    def generate_accessibility_report(self, session_id: str = None) -> Optional[Dict[str, str]]:
        """
        Generate accessibility report for specified or current session
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            Dictionary with paths to generated reports or None if failed
        """
        target_session = None
        
        if session_id:
            # Find session by ID
            for session in self.session_history:
                if session.session_id == session_id:
                    target_session = session
                    break
            if not target_session:
                logger.error(f"Session {session_id} not found")
                return None
        else:
            target_session = self.current_session
        
        if not target_session:
            logger.error("No session available for report generation")
            return None
        
        try:
            report_paths = self.report_generator.generate_comprehensive_report(target_session)
            logger.info(f"Generated accessibility report for session {target_session.session_id}")
            return report_paths
        except Exception as e:
            logger.error(f"Error generating accessibility report: {e}")
            return None
    
    def start_monitoring(self):
        """Start real-time accessibility monitoring"""
        self.monitoring_active = True
        self.violation_detector.enable_detection(True)
        logger.info("Started accessibility monitoring")
    
    def stop_monitoring(self):
        """Stop real-time accessibility monitoring"""
        self.monitoring_active = False
        self.violation_detector.enable_detection(False)
        logger.info("Stopped accessibility monitoring")
    
    def _get_test_configuration(self) -> Dict[str, Any]:
        """Get current test configuration"""
        return {
            "real_time_monitoring": self.config.enable_real_time_monitoring,
            "keyboard_testing": self.config.enable_keyboard_testing,
            "compliance_checking": self.config.enable_compliance_checking,
            "violation_detection": self.config.enable_violation_detection,
            "screenshot_capture": self.config.capture_screenshots,
            "compliance_level": self.config.compliance_level,
            "screenshot_dir": self.config.screenshot_dir,
            "report_dir": self.config.report_dir
        }
    
    def _violation_to_dict(self, violation: AccessibilityViolation) -> Dict[str, Any]:
        """Convert violation to dictionary for JSON serialization"""
        return {
            "rule_id": violation.rule_id,
            "severity": violation.severity.value,
            "description": violation.description,
            "location": violation.location,
            "screenshot_path": violation.screenshot_path,
            "suggestion": violation.suggestion,
            "wcag_reference": violation.wcag_reference,
            "section_508_reference": violation.section_508_reference
        }
    
    def _keyboard_test_to_dict(self, test: KeyboardTestCase) -> Dict[str, Any]:
        """Convert keyboard test to dictionary"""
        return {
            "test_id": test.test_id,
            "description": test.description,
            "keys_to_press": test.keys_to_press,
            "expected_behavior": test.expected_behavior,
            "result": test.result.value if test.result else None,
            "notes": test.notes
        }
    
    def _generate_recommendations(self, violations: List[AccessibilityViolation]) -> List[str]:
        """Generate accessibility recommendations based on violations"""
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            rule_type = violation.rule_id.split('_')[0]
            if rule_type not in violation_types:
                violation_types[rule_type] = 0
            violation_types[rule_type] += 1
        
        # Generate specific recommendations
        if violation_types.get('color', 0) > 0:
            recommendations.append(
                f"Fix {violation_types['color']} color contrast issues - ensure 4.5:1 ratio for normal text"
            )
        
        if violation_types.get('keyboard', 0) > 0:
            recommendations.append(
                f"Address {violation_types['keyboard']} keyboard accessibility issues - test tab navigation"
            )
        
        if violation_types.get('missing', 0) > 0:
            recommendations.append(
                f"Add {violation_types['missing']} missing text alternatives for images and content"
            )
        
        if violation_types.get('small', 0) > 0:
            recommendations.append(
                f"Increase {violation_types['small']} small click targets to minimum 44x44 pixels"
            )
        
        # Add general recommendations
        if len(violations) > 10:
            recommendations.append("Consider comprehensive accessibility audit with assistive technology users")
        
        if not recommendations:
            recommendations.append("Continue monitoring for accessibility issues")
        
        return recommendations[:5]  # Return top 5 recommendations