#!/usr/bin/env python3
"""
Accessibility Demo Script

Demonstrates the accessibility and 508 compliance testing capabilities
of Agent S without requiring a full agent setup.
"""

import io
import os
import sys
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Mock pyautogui for demo purposes
class MockPyAutoGUI:
    FAILSAFE = False
    def screenshot(self):
        # Create a mock screenshot with some accessibility issues
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some elements with accessibility issues
        # Low contrast text
        draw.rectangle([50, 50, 300, 100], fill='#e0e0e0')  # Light gray background
        draw.text((60, 65), "Low contrast text", fill='#c0c0c0')  # Very light gray text
        
        # Small click targets
        draw.rectangle([50, 120, 85, 140], fill='blue')  # 35x20 button (too small)
        
        # Missing alt text image simulation
        draw.rectangle([50, 160, 200, 260], fill='green')
        draw.text((60, 200), "Image without alt text", fill='white')
        
        # Good contrast example
        draw.rectangle([50, 280, 300, 330], fill='white', outline='black')
        draw.text((60, 295), "Good contrast text", fill='black')
        
        return img
    
    def press(self, key): pass
    def hotkey(self, *keys): pass
    def size(self): return (800, 600)

# Install mock
sys.modules['pyautogui'] = MockPyAutoGUI()

# Now import our accessibility modules
from gui_agents.s2_5.agents.accessibility_agent import AccessibilityAgent, AccessibilityConfig
from gui_agents.s2_5.accessibility.compliance_checker import ViolationSeverity

def print_banner():
    print("=" * 70)
    print("🔍 AGENT S ACCESSIBILITY DEMO")
    print("=" * 70)
    print("Demonstrating accessibility and 508 compliance testing capabilities")
    print()

def create_mock_observation():
    """Create a mock observation with screenshot"""
    mock_pyautogui = MockPyAutoGUI()
    screenshot = mock_pyautogui.screenshot()
    
    # Convert to bytes
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    screenshot_bytes = buffered.getvalue()
    
    return {"screenshot": screenshot_bytes}

def print_violations(violations):
    """Print violations in a formatted way"""
    if not violations:
        print("✅ No accessibility violations found!")
        return
    
    print(f"\n⚠️  Found {len(violations)} accessibility violations:")
    
    # Group by severity
    by_severity = {}
    for violation in violations:
        severity = violation.get('severity', 'unknown')
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(violation)
    
    # Print by severity
    severity_emojis = {
        'critical': '🚨',
        'major': '⚠️', 
        'minor': '💡',
        'warning': 'ℹ️'
    }
    
    for severity in ['critical', 'major', 'minor', 'warning']:
        if severity in by_severity:
            violations_list = by_severity[severity]
            print(f"\n{severity_emojis.get(severity, '•')} {severity.upper()} ({len(violations_list)} issues):")
            
            for i, violation in enumerate(violations_list, 1):
                desc = violation.get('description', 'Unknown violation')
                suggestion = violation.get('suggestion', '')
                wcag_ref = violation.get('wcag_reference', '')
                
                print(f"   {i}. {desc}")
                if suggestion:
                    print(f"      💡 {suggestion}")
                if wcag_ref:
                    print(f"      📖 {wcag_ref}")

def main():
    print_banner()
    
    try:
        # Configure accessibility testing
        print("🔧 Configuring accessibility testing...")
        config = AccessibilityConfig(
            enable_real_time_monitoring=True,
            enable_keyboard_testing=True,
            enable_compliance_checking=True,
            enable_violation_detection=True,
            capture_screenshots=True,
            compliance_level="AA",
            screenshot_dir="demo_screenshots",
            report_dir="demo_reports"
        )
        print(f"   ✓ Compliance Level: WCAG {config.compliance_level}")
        print(f"   ✓ Real-time Monitoring: {config.enable_real_time_monitoring}")
        print(f"   ✓ Keyboard Testing: {config.enable_keyboard_testing}")
        print(f"   ✓ Screenshot Capture: {config.capture_screenshots}")
        
        # Initialize accessibility agent
        print("\n🚀 Initializing accessibility agent...")
        accessibility_agent = AccessibilityAgent(config)
        print("   ✓ AccessibilityAgent initialized")
        
        # Start accessibility session
        print("\n📝 Starting accessibility testing session...")
        session_id = accessibility_agent.start_accessibility_session(
            "Demo Application", "linux"
        )
        print(f"   ✓ Session started: {session_id}")
        
        # Create mock observation
        print("\n📸 Capturing mock screenshot...")
        observation = create_mock_observation()
        print("   ✓ Screenshot captured (mock)")
        
        # Analyze for accessibility issues
        print("\n🔍 Analyzing accessibility compliance...")
        analysis = accessibility_agent.analyze_observation(observation)
        
        violations = analysis.get("violations", [])
        score = analysis.get("compliance_score", 0)
        recommendations = analysis.get("recommendations", [])
        
        print(f"   ✓ Analysis complete")
        print(f"   📊 Compliance Score: {score:.1f}/100")
        
        # Display violations
        print_violations(violations)
        
        # Show recommendations
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Run keyboard tests (mock)
        print(f"\n⌨️  Running keyboard accessibility tests...")
        try:
            keyboard_results = accessibility_agent.run_keyboard_accessibility_tests(observation)
            if keyboard_results and "summary" in keyboard_results:
                summary = keyboard_results["summary"]
                print(f"   ✓ Tests completed")
                print(f"   📊 Total Tests: {summary.get('total_tests', 0)}")
                print(f"   ✅ Passed: {summary.get('passed', 0)}")
                print(f"   ❌ Failed: {summary.get('failed', 0)}")
                print(f"   ⚠️  Warnings: {summary.get('warnings', 0)}")
                print(f"   ⏭️  Skipped: {summary.get('skipped', 0)}")
            else:
                print("   ⚠️  Keyboard tests not available in demo environment")
        except Exception as e:
            print(f"   ⚠️  Keyboard tests skipped: {str(e)}")
        
        # Get status
        print(f"\n📊 Getting accessibility status...")
        status = accessibility_agent.get_accessibility_status()
        current_session = status.get("current_session", {})
        print(f"   ✓ Session: {current_session.get('session_id', 'N/A')}")
        print(f"   📈 Violations: {current_session.get('violations_count', 0)}")
        print(f"   🔬 Tests: {current_session.get('keyboard_tests_count', 0)}")
        
        # End session and generate reports
        print(f"\n📄 Ending session and generating reports...")
        session_summary = accessibility_agent.end_accessibility_session()
        
        if session_summary:
            print(f"   ✓ Session completed")
            print(f"   📊 Final Score: {session_summary.compliance_score:.1f}/100")
            print(f"   🚨 Violations: {len(session_summary.violations)}")
            print(f"   ⌨️  Keyboard Tests: {len(session_summary.keyboard_tests)}")
            print(f"   📸 Evidence: {len(session_summary.evidence)}")
            
            # Try to generate reports
            try:
                report_paths = accessibility_agent.generate_accessibility_report()
                if report_paths:
                    print(f"\n📋 Reports generated:")
                    for report_type, path in report_paths.items():
                        if os.path.exists(path):
                            print(f"   📄 {report_type.upper()}: {path}")
                        else:
                            print(f"   📄 {report_type.upper()}: {path} (creation attempted)")
            except Exception as e:
                print(f"   ⚠️  Report generation encountered issues: {str(e)}")
        
        # Final summary
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"{'='*70}")
        
        if score >= 90:
            print("🌟 EXCELLENT accessibility compliance!")
        elif score >= 70:
            print("👍 GOOD accessibility compliance with room for improvement")
        elif score >= 50:
            print("⚠️  NEEDS IMPROVEMENT - significant accessibility barriers present")
        else:
            print("🚨 CRITICAL accessibility issues found - immediate attention required")
        
        print(f"\nThis demo showed:")
        print(f"• ✅ Accessibility compliance checking (Section 508 & WCAG)")
        print(f"• ✅ Real-time violation detection") 
        print(f"• ✅ Screenshot analysis and evidence capture")
        print(f"• ✅ Comprehensive reporting capabilities")
        print(f"• ✅ Integration with Agent S computer use framework")
        
        print(f"\nFor production use:")
        print(f"• Enable with: --enable_accessibility --accessibility_compliance_level AA")
        print(f"• Use standalone: python gui_agents/s2_5/accessibility/cli_accessibility_tester.py")
        print(f"• Check ACCESSIBILITY_README.md for full documentation")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())