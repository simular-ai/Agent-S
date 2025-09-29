#!/usr/bin/env python3
"""
Standalone Accessibility Testing CLI Tool

Command-line interface for running accessibility and 508 compliance tests
independently of the main Agent S framework.
"""

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

import pyautogui
from PIL import Image

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from gui_agents.s2_5.accessibility.accessibility_agent import AccessibilityAgent, AccessibilityConfig
from gui_agents.s2_5.accessibility.compliance_checker import ViolationSeverity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def capture_screenshot() -> bytes:
    """Capture current screen as bytes"""
    screenshot = pyautogui.screenshot()
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return buffered.getvalue()


def print_banner():
    """Print application banner"""
    print("=" * 70)
    print("🔍 AGENT S ACCESSIBILITY & 508 COMPLIANCE TESTER")
    print("=" * 70)
    print("Comprehensive accessibility testing for computer applications")
    print()


def print_violations_summary(violations: list):
    """Print summary of violations found"""
    if not violations:
        print("✅ No accessibility violations found!")
        return
    
    # Count by severity
    severity_counts = {}
    for violation in violations:
        severity = violation.get('severity', 'unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print(f"\n⚠️  Found {len(violations)} accessibility violations:")
    
    for severity in ['critical', 'major', 'minor', 'warning']:
        count = severity_counts.get(severity, 0)
        if count > 0:
            emoji = {"critical": "🚨", "major": "⚠️", "minor": "💡", "warning": "ℹ️"}
            print(f"   {emoji.get(severity, '•')} {severity.title()}: {count}")
    
    print("\nTop violations:")
    for i, violation in enumerate(violations[:5], 1):
        severity = violation.get('severity', 'unknown')
        description = violation.get('description', 'Unknown violation')
        print(f"   {i}. [{severity.upper()}] {description}")
    
    if len(violations) > 5:
        print(f"   ... and {len(violations) - 5} more")


def print_keyboard_test_results(test_results: Dict[str, Any]):
    """Print keyboard test results"""
    if not test_results or "test_results" not in test_results:
        print("❌ No keyboard test results available")
        return
    
    summary = test_results.get("summary", {})
    tests = test_results.get("test_results", [])
    
    print(f"\n⌨️  Keyboard Accessibility Test Results:")
    print(f"   • Total Tests: {summary.get('total_tests', 0)}")
    print(f"   • Passed: {summary.get('passed', 0)}")
    print(f"   • Failed: {summary.get('failed', 0)}")
    print(f"   • Warnings: {summary.get('warnings', 0)}")
    print(f"   • Skipped: {summary.get('skipped', 0)}")
    
    # Show failed tests
    failed_tests = [t for t in tests if t.get('result') == 'fail']
    if failed_tests:
        print(f"\n❌ Failed keyboard tests:")
        for test in failed_tests:
            print(f"   • {test.get('description', 'Unknown test')}")
            if test.get('notes'):
                print(f"     Notes: {test['notes']}")


def run_quick_scan(accessibility_agent: AccessibilityAgent, app_name: str) -> Dict[str, Any]:
    """Run a quick accessibility scan"""
    print("📸 Capturing screenshot...")
    screenshot_bytes = capture_screenshot()
    
    observation = {"screenshot": screenshot_bytes}
    
    print("🔍 Analyzing accessibility...")
    analysis = accessibility_agent.analyze_observation(observation)
    
    return analysis


def run_comprehensive_test(accessibility_agent: AccessibilityAgent, app_name: str) -> Dict[str, Any]:
    """Run comprehensive accessibility testing"""
    print("🚀 Starting comprehensive accessibility testing...")
    
    # Start session
    session_id = accessibility_agent.start_accessibility_session(app_name)
    print(f"📝 Session ID: {session_id}")
    
    results = {}
    
    try:
        # Capture initial screenshot
        print("\n📸 Capturing screenshot...")
        screenshot_bytes = capture_screenshot()
        observation = {"screenshot": screenshot_bytes}
        
        # Run accessibility analysis
        print("🔍 Running accessibility compliance checks...")
        analysis = accessibility_agent.analyze_observation(observation)
        results["accessibility_analysis"] = analysis
        
        # Run keyboard tests
        print("⌨️  Running keyboard accessibility tests...")
        keyboard_results = accessibility_agent.run_keyboard_accessibility_tests(observation)
        results["keyboard_tests"] = keyboard_results
        
        # Wait a moment for any additional violations to be detected
        print("⏳ Monitoring for additional issues...")
        time.sleep(2)
        
        # Get final status
        status = accessibility_agent.get_accessibility_status()
        results["final_status"] = status
        
    finally:
        # End session and generate reports
        print("📊 Generating reports...")
        session_summary = accessibility_agent.end_accessibility_session()
        results["session_summary"] = session_summary
        
        if session_summary:
            report_paths = accessibility_agent.generate_accessibility_report()
            results["report_paths"] = report_paths
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Accessibility & 508 Compliance Tester"
    )
    
    parser.add_argument(
        "--app-name",
        type=str,
        default="Current Application",
        help="Name of the application being tested"
    )
    
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["quick", "comprehensive"],
        default="quick",
        help="Type of accessibility test to run"
    )
    
    parser.add_argument(
        "--compliance-level",
        type=str,
        choices=["AA", "AAA"],
        default="AA",
        help="WCAG compliance level to test against"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="accessibility_results",
        help="Directory to save results and reports"
    )
    
    parser.add_argument(
        "--enable-keyboard-tests",
        action="store_true",
        default=True,
        help="Enable keyboard navigation testing"
    )
    
    parser.add_argument(
        "--enable-screenshots",
        action="store_true", 
        default=True,
        help="Capture screenshots for violations"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure accessibility testing
    config = AccessibilityConfig(
        enable_real_time_monitoring=True,
        enable_keyboard_testing=args.enable_keyboard_tests,
        enable_compliance_checking=True,
        enable_violation_detection=True,
        capture_screenshots=args.enable_screenshots,
        compliance_level=args.compliance_level,
        screenshot_dir=os.path.join(args.output_dir, "screenshots"),
        report_dir=os.path.join(args.output_dir, "reports")
    )
    
    # Initialize accessibility agent
    print(f"🔧 Initializing accessibility testing (WCAG {args.compliance_level})...")
    accessibility_agent = AccessibilityAgent(config)
    
    try:
        if args.test_type == "quick":
            print(f"⚡ Running quick accessibility scan for: {args.app_name}")
            results = run_quick_scan(accessibility_agent, args.app_name)
            
            # Display results
            violations = results.get("violations", [])
            print_violations_summary(violations)
            
            score = results.get("compliance_score", 0)
            print(f"\n📊 Accessibility Score: {score:.1f}/100")
            
            if score >= 90:
                print("🎉 Excellent accessibility compliance!")
            elif score >= 70:
                print("👍 Good accessibility compliance with room for improvement")
            elif score >= 50:
                print("⚠️  Needs significant accessibility improvements")
            else:
                print("🚨 Critical accessibility issues found")
                
        else:  # comprehensive
            print(f"🔍 Running comprehensive accessibility test for: {args.app_name}")
            results = run_comprehensive_test(accessibility_agent, args.app_name)
            
            # Display accessibility analysis
            analysis = results.get("accessibility_analysis", {})
            violations = analysis.get("violations", [])
            print_violations_summary(violations)
            
            # Display keyboard test results
            keyboard_tests = results.get("keyboard_tests", {})
            print_keyboard_test_results(keyboard_tests)
            
            # Display session summary
            session_summary = results.get("session_summary", {})
            if session_summary:
                print(f"\n📊 Final Results:")
                print(f"   • Total Violations: {session_summary.get('violations_found', 0)}")
                print(f"   • Compliance Score: {session_summary.get('compliance_score', 0):.1f}/100")
                print(f"   • Keyboard Tests: {session_summary.get('keyboard_tests_run', 0)}")
                print(f"   • Evidence Captured: {session_summary.get('evidence_captured', 0)}")
            
            # Display report paths
            report_paths = results.get("report_paths", {})
            if report_paths:
                print(f"\n📄 Reports Generated:")
                for report_type, path in report_paths.items():
                    print(f"   • {report_type.upper()}: {path}")
                    
                print(f"\nOpen the HTML report in your browser to view detailed results:")
                html_path = report_paths.get("html")
                if html_path:
                    print(f"file://{os.path.abspath(html_path)}")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during accessibility testing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print(f"\n✅ Accessibility testing complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()