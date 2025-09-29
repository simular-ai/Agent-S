"""
Accessibility Report Generator

Generates comprehensive accessibility and 508 compliance reports including:
- Executive summary
- Detailed violation listings
- Screenshot evidence
- Remediation recommendations
- WCAG/508 compliance scoring
- Export to multiple formats (HTML, PDF, JSON, CSV)
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from jinja2 import Template
import base64

from .compliance_checker import AccessibilityViolation, ViolationSeverity
from .keyboard_navigator import KeyboardTestCase, KeyboardTestResult
from .violation_detector import ViolationEvidence

logger = logging.getLogger(__name__)


@dataclass
class AccessibilityTestSession:
    """Complete accessibility test session data"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    url_or_app: str
    platform: str
    violations: List[AccessibilityViolation]
    keyboard_tests: List[KeyboardTestCase]
    evidence: List[ViolationEvidence]
    compliance_score: float
    test_configuration: Dict[str, Any]


class AccessibilityReportGenerator:
    """
    Comprehensive accessibility report generator
    """
    
    def __init__(self, output_dir: str = "accessibility_reports"):
        self.output_dir = output_dir
        self.ensure_output_directory()
        
        # Report templates
        self.html_template = self._get_html_template()
        self.json_template = self._get_json_template()
        
    def ensure_output_directory(self):
        """Ensure output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["html", "json", "csv", "screenshots", "assets"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def generate_comprehensive_report(self, test_session: AccessibilityTestSession) -> Dict[str, str]:
        """
        Generate comprehensive accessibility report in multiple formats
        
        Args:
            test_session: Complete test session data
            
        Returns:
            Dictionary with paths to generated reports
        """
        logger.info(f"Generating comprehensive accessibility report for session {test_session.session_id}")
        
        report_paths = {}
        
        # Generate HTML report
        html_path = self.generate_html_report(test_session)
        if html_path:
            report_paths["html"] = html_path
        
        # Generate JSON report
        json_path = self.generate_json_report(test_session)
        if json_path:
            report_paths["json"] = json_path
        
        # Generate CSV report
        csv_path = self.generate_csv_report(test_session)
        if csv_path:
            report_paths["csv"] = csv_path
        
        # Generate executive summary
        summary_path = self.generate_executive_summary(test_session)
        if summary_path:
            report_paths["executive_summary"] = summary_path
        
        logger.info(f"Generated {len(report_paths)} report formats")
        return report_paths
    
    def generate_html_report(self, test_session: AccessibilityTestSession) -> str:
        """Generate detailed HTML report"""
        try:
            # Prepare data for template
            template_data = self._prepare_template_data(test_session)
            
            # Render HTML template
            html_content = self.html_template.render(**template_data)
            
            # Save HTML file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accessibility_report_{test_session.session_id}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, "html", filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return None
    
    def generate_json_report(self, test_session: AccessibilityTestSession) -> str:
        """Generate JSON report for programmatic access"""
        try:
            # Convert test session to dictionary
            report_data = {
                "metadata": {
                    "session_id": test_session.session_id,
                    "generated_at": datetime.now().isoformat(),
                    "start_time": test_session.start_time.isoformat(),
                    "end_time": test_session.end_time.isoformat() if test_session.end_time else None,
                    "url_or_app": test_session.url_or_app,
                    "platform": test_session.platform,
                    "test_configuration": test_session.test_configuration
                },
                "summary": {
                    "compliance_score": test_session.compliance_score,
                    "total_violations": len(test_session.violations),
                    "critical_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.CRITICAL]),
                    "major_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.MAJOR]),
                    "minor_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.MINOR]),
                    "warning_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.WARNING]),
                    "keyboard_tests_passed": len([t for t in test_session.keyboard_tests if t.result == KeyboardTestResult.PASS]),
                    "keyboard_tests_failed": len([t for t in test_session.keyboard_tests if t.result == KeyboardTestResult.FAIL]),
                    "evidence_captured": len(test_session.evidence)
                },
                "violations": [self._violation_to_dict(v) for v in test_session.violations],
                "keyboard_tests": [self._keyboard_test_to_dict(t) for t in test_session.keyboard_tests],
                "evidence": [self._evidence_to_dict(e) for e in test_session.evidence]
            }
            
            # Save JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accessibility_report_{test_session.session_id}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, "json", filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return None
    
    def generate_csv_report(self, test_session: AccessibilityTestSession) -> str:
        """Generate CSV report for spreadsheet analysis"""
        try:
            import csv
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accessibility_violations_{test_session.session_id}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, "csv", filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'Rule ID', 'Severity', 'Description', 'WCAG Reference', 
                    '508 Reference', 'Location X', 'Location Y', 'Width', 'Height',
                    'Suggestion', 'Screenshot Path'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for violation in test_session.violations:
                    row = {
                        'Rule ID': violation.rule_id,
                        'Severity': violation.severity.value,
                        'Description': violation.description,
                        'WCAG Reference': violation.wcag_reference or '',
                        '508 Reference': violation.section_508_reference or '',
                        'Location X': violation.location.get('x', '') if violation.location else '',
                        'Location Y': violation.location.get('y', '') if violation.location else '',
                        'Width': violation.location.get('width', '') if violation.location else '',
                        'Height': violation.location.get('height', '') if violation.location else '',
                        'Suggestion': violation.suggestion or '',
                        'Screenshot Path': violation.screenshot_path or ''
                    }
                    writer.writerow(row)
            
            logger.info(f"CSV report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")
            return None
    
    def generate_executive_summary(self, test_session: AccessibilityTestSession) -> str:
        """Generate executive summary report"""
        try:
            summary_data = self._generate_summary_statistics(test_session)
            
            # Create executive summary template
            summary_template = Template("""
# Accessibility Testing Executive Summary

## Test Overview
- **Application/URL**: {{ url_or_app }}
- **Test Date**: {{ test_date }}
- **Platform**: {{ platform }}
- **Session ID**: {{ session_id }}

## Compliance Score: {{ compliance_score }}/100

## Key Findings
- **Total Violations**: {{ total_violations }}
- **Critical Issues**: {{ critical_violations }}
- **Major Issues**: {{ major_violations }}
- **Minor Issues**: {{ minor_violations }}

## Keyboard Accessibility
- **Tests Passed**: {{ keyboard_passed }}/{{ keyboard_total }}
- **Tests Failed**: {{ keyboard_failed }}/{{ keyboard_total }}

## Priority Recommendations
{% for rec in priority_recommendations %}
{{ loop.index }}. {{ rec }}
{% endfor %}

## Compliance Status
{% if compliance_score >= 90 %}
✅ **EXCELLENT** - High level of accessibility compliance
{% elif compliance_score >= 70 %}
⚠️ **GOOD** - Generally accessible with some improvements needed
{% elif compliance_score >= 50 %}
❌ **NEEDS IMPROVEMENT** - Significant accessibility barriers present
{% else %}
🚨 **CRITICAL** - Major accessibility barriers preventing access
{% endif %}

## Next Steps
1. Address all critical violations immediately
2. Implement keyboard navigation fixes
3. Review and test color contrast issues
4. Validate with assistive technology users
5. Re-test after remediation

---
Generated on {{ generated_date }} by Agent S Accessibility Testing System
            """)
            
            summary_content = summary_template.render(**summary_data)
            
            # Save summary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"executive_summary_{test_session.session_id}_{timestamp}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            logger.info(f"Executive summary generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return None
    
    def _prepare_template_data(self, test_session: AccessibilityTestSession) -> Dict[str, Any]:
        """Prepare data for HTML template rendering"""
        
        # Group violations by severity
        violations_by_severity = {
            'critical': [v for v in test_session.violations if v.severity == ViolationSeverity.CRITICAL],
            'major': [v for v in test_session.violations if v.severity == ViolationSeverity.MAJOR],
            'minor': [v for v in test_session.violations if v.severity == ViolationSeverity.MINOR],
            'warning': [v for v in test_session.violations if v.severity == ViolationSeverity.WARNING]
        }
        
        # Group violations by WCAG principle
        violations_by_principle = self._group_violations_by_principle(test_session.violations)
        
        # Calculate statistics
        stats = self._generate_summary_statistics(test_session)
        
        return {
            "session": test_session,
            "violations_by_severity": violations_by_severity,
            "violations_by_principle": violations_by_principle,
            "stats": stats,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "css_styles": self._get_css_styles(),
            "javascript": self._get_javascript()
        }
    
    def _group_violations_by_principle(self, violations: List[AccessibilityViolation]) -> Dict[str, List]:
        """Group violations by WCAG principle"""
        principles = {
            "Perceivable": [],
            "Operable": [], 
            "Understandable": [],
            "Robust": [],
            "Other": []
        }
        
        # Simple mapping based on WCAG references
        principle_mapping = {
            "1.": "Perceivable",
            "2.": "Operable", 
            "3.": "Understandable",
            "4.": "Robust"
        }
        
        for violation in violations:
            wcag_ref = violation.wcag_reference or ""
            principle = "Other"
            
            for prefix, principle_name in principle_mapping.items():
                if prefix in wcag_ref:
                    principle = principle_name
                    break
            
            principles[principle].append(violation)
        
        return principles
    
    def _generate_summary_statistics(self, test_session: AccessibilityTestSession) -> Dict[str, Any]:
        """Generate summary statistics for the test session"""
        
        keyboard_passed = len([t for t in test_session.keyboard_tests if t.result == KeyboardTestResult.PASS])
        keyboard_failed = len([t for t in test_session.keyboard_tests if t.result == KeyboardTestResult.FAIL])
        keyboard_total = len(test_session.keyboard_tests)
        
        # Generate priority recommendations based on violations
        priority_recommendations = self._generate_priority_recommendations(test_session.violations)
        
        return {
            "session_id": test_session.session_id,
            "url_or_app": test_session.url_or_app,
            "platform": test_session.platform,
            "test_date": test_session.start_time.strftime("%Y-%m-%d"),
            "compliance_score": round(test_session.compliance_score, 1),
            "total_violations": len(test_session.violations),
            "critical_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.CRITICAL]),
            "major_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.MAJOR]),
            "minor_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.MINOR]),
            "warning_violations": len([v for v in test_session.violations if v.severity == ViolationSeverity.WARNING]),
            "keyboard_passed": keyboard_passed,
            "keyboard_failed": keyboard_failed,
            "keyboard_total": keyboard_total,
            "evidence_count": len(test_session.evidence),
            "priority_recommendations": priority_recommendations,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _generate_priority_recommendations(self, violations: List[AccessibilityViolation]) -> List[str]:
        """Generate priority recommendations based on violations"""
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            rule_type = violation.rule_id.split('_')[0]
            if rule_type not in violation_types:
                violation_types[rule_type] = []
            violation_types[rule_type].append(violation)
        
        # Generate recommendations based on most common/severe issues
        if 'color' in violation_types:
            recommendations.append("Improve color contrast ratios to meet WCAG AA standards (4.5:1 minimum)")
        
        if 'keyboard' in violation_types:
            recommendations.append("Ensure all interactive elements are keyboard accessible and have focus indicators")
        
        if 'missing' in violation_types:
            recommendations.append("Add text alternatives for all images and non-text content")
        
        if 'small' in violation_types:
            recommendations.append("Increase click target sizes to minimum 44x44 pixels")
        
        # Add generic recommendations if no specific ones
        if not recommendations:
            recommendations.append("Review and address all identified accessibility violations")
            recommendations.append("Test with screen readers and other assistive technologies")
        
        return recommendations[:5]  # Return top 5 recommendations
    
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
        """Convert keyboard test to dictionary for JSON serialization"""
        return {
            "test_id": test.test_id,
            "description": test.description,
            "keys_to_press": test.keys_to_press,
            "expected_behavior": test.expected_behavior,
            "result": test.result.value if test.result else None,
            "notes": test.notes
        }
    
    def _evidence_to_dict(self, evidence: ViolationEvidence) -> Dict[str, Any]:
        """Convert evidence to dictionary for JSON serialization"""
        return {
            "violation_id": evidence.violation_id,
            "timestamp": evidence.timestamp,
            "screenshot_path": evidence.screenshot_path,
            "description": evidence.description,
            "coordinates": evidence.coordinates,
            "additional_data": evidence.additional_data
        }
    
    def _get_html_template(self) -> Template:
        """Get HTML report template"""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessibility Testing Report - {{ session.session_id }}</title>
    <style>{{ css_styles }}</style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>🔍 Accessibility Testing Report</h1>
            <div class="meta-info">
                <p><strong>Application/URL:</strong> {{ session.url_or_app }}</p>
                <p><strong>Platform:</strong> {{ session.platform }}</p>
                <p><strong>Test Date:</strong> {{ session.start_time.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Session ID:</strong> {{ session.session_id }}</p>
            </div>
        </header>

        <section class="summary">
            <h2>📊 Executive Summary</h2>
            <div class="score-card">
                <div class="score">
                    <span class="score-value {{ 'excellent' if stats.compliance_score >= 90 else 'good' if stats.compliance_score >= 70 else 'needs-improvement' if stats.compliance_score >= 50 else 'critical' }}">
                        {{ stats.compliance_score }}/100
                    </span>
                    <span class="score-label">Compliance Score</span>
                </div>
                <div class="stats-grid">
                    <div class="stat">
                        <span class="stat-value">{{ stats.total_violations }}</span>
                        <span class="stat-label">Total Violations</span>
                    </div>
                    <div class="stat critical">
                        <span class="stat-value">{{ stats.critical_violations }}</span>
                        <span class="stat-label">Critical</span>
                    </div>
                    <div class="stat major">
                        <span class="stat-value">{{ stats.major_violations }}</span>
                        <span class="stat-label">Major</span>
                    </div>
                    <div class="stat minor">
                        <span class="stat-value">{{ stats.minor_violations }}</span>
                        <span class="stat-label">Minor</span>
                    </div>
                </div>
            </div>
        </section>

        <section class="keyboard-tests">
            <h2>⌨️ Keyboard Accessibility Tests</h2>
            <div class="test-results">
                {% for test in session.keyboard_tests %}
                <div class="test-result {{ test.result.value if test.result else 'unknown' }}">
                    <h3>{{ test.description }}</h3>
                    <p><strong>Result:</strong> <span class="result-badge {{ test.result.value if test.result else 'unknown' }}">{{ test.result.value.upper() if test.result else 'UNKNOWN' }}</span></p>
                    <p><strong>Expected:</strong> {{ test.expected_behavior }}</p>
                    {% if test.notes %}
                    <p><strong>Notes:</strong> {{ test.notes }}</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </section>

        <section class="violations">
            <h2>🚨 Accessibility Violations</h2>
            
            {% for severity, violation_list in violations_by_severity.items() %}
            {% if violation_list %}
            <div class="severity-section">
                <h3 class="severity-header {{ severity }}">
                    {{ severity.title() }} Issues ({{ violation_list|length }})
                </h3>
                
                {% for violation in violation_list %}
                <div class="violation-card {{ severity }}">
                    <div class="violation-header">
                        <h4>{{ violation.rule_id }}</h4>
                        <span class="severity-badge {{ severity }}">{{ severity.title() }}</span>
                    </div>
                    
                    <p class="violation-description">{{ violation.description }}</p>
                    
                    {% if violation.location %}
                    <p class="violation-location">
                        <strong>Location:</strong> ({{ violation.location.x }}, {{ violation.location.y }}) 
                        {{ violation.location.width }}×{{ violation.location.height }}px
                    </p>
                    {% endif %}
                    
                    {% if violation.suggestion %}
                    <div class="violation-suggestion">
                        <strong>💡 Suggestion:</strong> {{ violation.suggestion }}
                    </div>
                    {% endif %}
                    
                    <div class="violation-references">
                        {% if violation.wcag_reference %}
                        <span class="reference wcag">{{ violation.wcag_reference }}</span>
                        {% endif %}
                        {% if violation.section_508_reference %}
                        <span class="reference section508">Section 508: {{ violation.section_508_reference }}</span>
                        {% endif %}
                    </div>
                    
                    {% if violation.screenshot_path %}
                    <div class="violation-evidence">
                        <img src="{{ violation.screenshot_path }}" alt="Screenshot evidence for {{ violation.rule_id }}" class="evidence-screenshot">
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            {% endfor %}
        </section>

        <footer class="report-footer">
            <p>Report generated on {{ generated_date }} by Agent S Accessibility Testing System</p>
        </footer>
    </div>

    <script>{{ javascript }}</script>
</body>
</html>
        """
        return Template(template_str)
    
    def _get_json_template(self) -> Template:
        """Get JSON report template (for structured data export)"""
        # JSON template is handled directly in generate_json_report
        return Template("{}")
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .report-header {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .report-header h1 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 2.5em;
        }
        
        .meta-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }
        
        .summary {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .score-card {
            display: flex;
            align-items: center;
            gap: 40px;
        }
        
        .score {
            text-align: center;
            min-width: 150px;
        }
        
        .score-value {
            display: block;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .score-value.excellent { color: #27ae60; }
        .score-value.good { color: #f39c12; }
        .score-value.needs-improvement { color: #e67e22; }
        .score-value.critical { color: #e74c3c; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            flex: 1;
        }
        
        .stat {
            text-align: center;
            padding: 15px;
            border-radius: 6px;
            background: #ecf0f1;
        }
        
        .stat.critical { background: #ffebee; }
        .stat.major { background: #fff3e0; }
        .stat.minor { background: #e8f5e8; }
        
        .stat-value {
            display: block;
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .keyboard-tests, .violations {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .test-result {
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .test-result.pass { border-left: 4px solid #27ae60; }
        .test-result.fail { border-left: 4px solid #e74c3c; }
        .test-result.warning { border-left: 4px solid #f39c12; }
        .test-result.skip { border-left: 4px solid #95a5a6; }
        
        .result-badge {
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .result-badge.pass { background: #d5f4e6; color: #27ae60; }
        .result-badge.fail { background: #fdeaea; color: #e74c3c; }
        .result-badge.warning { background: #fef9e7; color: #f39c12; }
        .result-badge.skip { background: #ecf0f1; color: #95a5a6; }
        
        .severity-section {
            margin-bottom: 30px;
        }
        
        .severity-header {
            padding: 15px;
            border-radius: 6px 6px 0 0;
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .severity-header.critical { background: #e74c3c; color: white; }
        .severity-header.major { background: #e67e22; color: white; }
        .severity-header.minor { background: #f39c12; color: white; }
        .severity-header.warning { background: #95a5a6; color: white; }
        
        .violation-card {
            border: 1px solid #ddd;
            border-top: none;
            padding: 20px;
            margin-bottom: 1px;
        }
        
        .violation-card:last-child {
            border-radius: 0 0 6px 6px;
        }
        
        .violation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .severity-badge {
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .severity-badge.critical { background: #e74c3c; color: white; }
        .severity-badge.major { background: #e67e22; color: white; }
        .severity-badge.minor { background: #f39c12; color: white; }
        .severity-badge.warning { background: #95a5a6; color: white; }
        
        .violation-suggestion {
            background: #e8f5e8;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .violation-references {
            margin: 15px 0;
        }
        
        .reference {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-right: 8px;
        }
        
        .reference.wcag { background: #3498db; color: white; }
        .reference.section508 { background: #9b59b6; color: white; }
        
        .evidence-screenshot {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 15px;
        }
        
        .report-footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .score-card {
                flex-direction: column;
                gap: 20px;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .meta-info {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for HTML report interactivity"""
        return """
        // Add interactive features to the report
        document.addEventListener('DOMContentLoaded', function() {
            // Add click to expand functionality for violation cards
            const violationCards = document.querySelectorAll('.violation-card');
            violationCards.forEach(card => {
                const evidenceSection = card.querySelector('.violation-evidence');
                if (evidenceSection) {
                    const screenshot = evidenceSection.querySelector('.evidence-screenshot');
                    if (screenshot) {
                        screenshot.style.cursor = 'pointer';
                        screenshot.addEventListener('click', function() {
                            // Open screenshot in a modal or new window
                            window.open(this.src, '_blank');
                        });
                    }
                }
            });
            
            // Add filtering functionality
            const addFilterButtons = () => {
                const container = document.querySelector('.violations');
                if (!container) return;
                
                const filterDiv = document.createElement('div');
                filterDiv.className = 'filter-controls';
                filterDiv.style.marginBottom = '20px';
                
                const severities = ['all', 'critical', 'major', 'minor', 'warning'];
                severities.forEach(severity => {
                    const button = document.createElement('button');
                    button.textContent = severity.charAt(0).toUpperCase() + severity.slice(1);
                    button.className = `filter-btn ${severity === 'all' ? 'active' : ''}`;
                    button.style.marginRight = '10px';
                    button.style.padding = '8px 16px';
                    button.style.border = '1px solid #ddd';
                    button.style.borderRadius = '4px';
                    button.style.background = severity === 'all' ? '#3498db' : 'white';
                    button.style.color = severity === 'all' ? 'white' : '#333';
                    button.style.cursor = 'pointer';
                    
                    button.addEventListener('click', function() {
                        // Update active button
                        document.querySelectorAll('.filter-btn').forEach(btn => {
                            btn.style.background = 'white';
                            btn.style.color = '#333';
                            btn.classList.remove('active');
                        });
                        this.style.background = '#3498db';
                        this.style.color = 'white';
                        this.classList.add('active');
                        
                        // Filter violations
                        const sections = document.querySelectorAll('.severity-section');
                        sections.forEach(section => {
                            if (severity === 'all') {
                                section.style.display = 'block';
                            } else {
                                const header = section.querySelector('.severity-header');
                                if (header && header.classList.contains(severity)) {
                                    section.style.display = 'block';
                                } else {
                                    section.style.display = 'none';
                                }
                            }
                        });
                    });
                    
                    filterDiv.appendChild(button);
                });
                
                container.insertBefore(filterDiv, container.firstChild.nextSibling);
            };
            
            addFilterButtons();
        });
        """