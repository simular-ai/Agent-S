# Accessibility and 508 Compliance Testing with Agent S

## Overview

Agent S now includes comprehensive accessibility and Section 508 compliance testing capabilities. This integration provides real-time accessibility monitoring, keyboard navigation testing, violation detection, and comprehensive reporting during computer use automation.

## Features

### 🔍 **Comprehensive Compliance Testing**
- Section 508 compliance checking
- WCAG 2.1 AA and AAA standard support
- Real-time violation detection
- Color contrast analysis
- Focus indicator validation

### ⌨️ **Keyboard Navigation Testing**
- Tab order validation
- Focus management testing
- Keyboard shortcuts verification
- Skip links functionality
- Modal dialog keyboard trapping

### 📊 **Advanced Reporting**
- HTML reports with interactive features
- JSON reports for programmatic analysis
- CSV exports for spreadsheet analysis
- Executive summaries with actionable recommendations
- Screenshot evidence with violation highlighting

### 🚨 **Real-time Monitoring**
- Live accessibility monitoring during agent execution
- Automatic violation detection and evidence capture
- Integration with existing Agent S workflow
- Configurable compliance levels and testing options

## Quick Start

### 1. Enable Accessibility in Agent S

```bash
agent_s \
    --provider openai \
    --model gpt-5-2025-08-07 \
    --ground_provider huggingface \
    --ground_url http://localhost:8080 \
    --ground_model ui-tars-1.5-7b \
    --grounding_width 1920 \
    --grounding_height 1080 \
    --enable_accessibility \
    --accessibility_compliance_level AA \
    --accessibility_screenshots \
    --accessibility_reports_dir ./reports
```

### 2. Standalone Accessibility Testing

For quick accessibility testing of any application:

```bash
# Quick scan
python gui_agents/s2_5/accessibility/cli_accessibility_tester.py \
    --app-name "My Application" \
    --test-type quick

# Comprehensive testing
python gui_agents/s2_5/accessibility/cli_accessibility_tester.py \
    --app-name "My Application" \
    --test-type comprehensive \
    --compliance-level AA \
    --enable-keyboard-tests \
    --enable-screenshots \
    --output-dir ./accessibility_results
```

### 3. Programmatic Usage

```python
from gui_agents.s2_5.agents.agent_s import AgentS2_5
from gui_agents.s2_5.agents.accessibility_agent import AccessibilityConfig
from gui_agents.s2_5.agents.grounding import OSWorldACI

# Configure accessibility testing
accessibility_config = AccessibilityConfig(
    enable_real_time_monitoring=True,
    enable_keyboard_testing=True,
    enable_compliance_checking=True,
    capture_screenshots=True,
    compliance_level="AA",  # or "AAA"
    report_dir="./accessibility_reports"
)

# Initialize agent with accessibility
agent = AgentS2_5(
    engine_params=engine_params,
    grounding_agent=grounding_agent,
    enable_accessibility=True,
    accessibility_config=accessibility_config
)

# Start accessibility session
session_id = agent.start_accessibility_session("My Application")

# Normal agent operation with automatic accessibility monitoring
obs = {"screenshot": screenshot_bytes}
info, actions = agent.predict("Open the settings menu", obs)

# Check for accessibility violations
if "accessibility_violations" in info:
    violations = info["accessibility_violations"]
    print(f"Found {len(violations)} accessibility violations")

# Run comprehensive keyboard tests
keyboard_results = agent.run_keyboard_accessibility_tests(obs)
print(f"Keyboard tests: {keyboard_results['summary']}")

# End session and generate reports
session_summary = agent.end_accessibility_session()
report_paths = agent.generate_accessibility_report()

print(f"Accessibility Score: {session_summary['compliance_score']:.1f}/100")
print(f"HTML Report: {report_paths['html']}")
```

## Configuration Options

### AccessibilityConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_real_time_monitoring` | bool | True | Enable live monitoring during agent execution |
| `enable_keyboard_testing` | bool | True | Enable comprehensive keyboard navigation testing |
| `enable_compliance_checking` | bool | True | Enable Section 508 and WCAG compliance checking |
| `enable_violation_detection` | bool | True | Enable real-time violation detection |
| `capture_screenshots` | bool | True | Capture screenshots for violation evidence |
| `compliance_level` | str | "AA" | WCAG compliance level ("AA" or "AAA") |
| `screenshot_dir` | str | "accessibility_screenshots" | Directory for violation screenshots |
| `report_dir` | str | "accessibility_reports" | Directory for generated reports |
| `auto_generate_reports` | bool | True | Automatically generate reports at session end |

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--enable_accessibility` | Enable accessibility testing in Agent S |
| `--accessibility_monitoring` | Enable real-time monitoring |
| `--accessibility_keyboard_tests` | Enable keyboard navigation testing |
| `--accessibility_compliance_level` | Set compliance level (AA or AAA) |
| `--accessibility_screenshots` | Enable screenshot capture |
| `--accessibility_reports_dir` | Directory for reports |

## Understanding Reports

### HTML Reports
Interactive HTML reports include:
- Executive summary with compliance score
- Detailed violation listings by severity
- Keyboard test results
- Screenshot evidence with highlighting
- Filterable violation views
- WCAG and Section 508 references

### JSON Reports
Structured data for programmatic analysis:
```json
{
  "metadata": {
    "session_id": "accessibility_session_123",
    "compliance_score": 75.5,
    "total_violations": 12
  },
  "violations": [
    {
      "rule_id": "color_contrast_violation",
      "severity": "major",
      "description": "Low color contrast detected",
      "wcag_reference": "WCAG 2.1 SC 1.4.3",
      "suggestion": "Increase contrast ratio to 4.5:1"
    }
  ],
  "keyboard_tests": [...],
  "evidence": [...]
}
```

### Compliance Scoring

| Score Range | Status | Description |
|-------------|--------|-------------|
| 90-100 | 🎉 Excellent | High level of accessibility compliance |
| 70-89 | 👍 Good | Generally accessible with minor improvements needed |
| 50-69 | ⚠️ Needs Improvement | Significant accessibility barriers present |
| 0-49 | 🚨 Critical | Major accessibility barriers preventing access |

## Violation Types

### Critical Violations
- Missing text alternatives for images
- Elements not keyboard accessible
- Critical color contrast failures
- Missing form labels

### Major Violations
- Poor color contrast (below AA standards)
- Missing focus indicators
- Heading structure issues
- Table accessibility problems

### Minor Violations
- Small click targets (below 44x44px)
- Minor color contrast issues
- Non-descriptive link text

### Warnings
- Potential keyboard traps
- Best practice recommendations

## Integration with Existing Workflows

### OSWorld Integration
The accessibility system integrates seamlessly with OSWorld testing:

```python
# In your OSWorld test script
from gui_agents.s2_5.agents.agent_s import AgentS2_5
from gui_agents.s2_5.agents.accessibility_agent import AccessibilityConfig

# Enable accessibility for OSWorld tests
agent = AgentS2_5(
    engine_params=engine_params,
    grounding_agent=osworld_grounding_agent,
    enable_accessibility=True,
    accessibility_config=AccessibilityConfig(compliance_level="AA")
)

# Accessibility will be monitored throughout the test
```

### Custom Integrations
Create custom accessibility workflows:

```python
from gui_agents.s2_5.accessibility import (
    AccessibilityAgent,
    AccessibilityConfig,
    ViolationDetector
)

# Custom accessibility testing workflow
config = AccessibilityConfig(
    enable_real_time_monitoring=True,
    compliance_level="AAA"
)

accessibility_agent = AccessibilityAgent(config)
session_id = accessibility_agent.start_accessibility_session("Custom App")

# Your custom testing logic here
observation = {"screenshot": get_screenshot()}
analysis = accessibility_agent.analyze_observation(observation)

# Generate custom reports
reports = accessibility_agent.generate_accessibility_report()
```

## Best Practices

### 1. Testing Strategy
- Start with quick scans to identify major issues
- Use comprehensive testing for detailed analysis
- Test keyboard navigation separately for thorough coverage
- Monitor compliance during development cycles

### 2. Violation Prioritization
1. **Critical violations first** - These prevent access to content
2. **Major violations** - Significantly impact usability
3. **Minor violations** - Improve overall experience
4. **Warnings** - Address as best practices

### 3. Evidence Collection
- Enable screenshot capture for clear violation documentation
- Review HTML reports for detailed analysis
- Use JSON exports for automated processing
- Share executive summaries with stakeholders

### 4. Continuous Monitoring
- Enable real-time monitoring during development
- Set up automated accessibility testing in CI/CD
- Regular compliance reviews with generated reports
- Track improvement over time with scoring

## Troubleshooting

### Common Issues

**Q: Accessibility testing not working in headless environments**
A: Accessibility testing requires a display for screenshot capture. Use Xvfb or similar for headless testing.

**Q: Keyboard tests failing unexpectedly**
A: Ensure the application has focus and no modal dialogs are blocking interaction.

**Q: False positive color contrast violations**
A: The system uses conservative detection. Review screenshots to verify actual violations.

**Q: Reports not generating**
A: Check write permissions for the report directory and ensure all dependencies are installed.

### Debug Mode
Enable verbose logging for debugging:

```python
import logging
logging.getLogger('gui_agents.s2_5.accessibility').setLevel(logging.DEBUG)
```

Or use the CLI with verbose flag:
```bash
python gui_agents/s2_5/accessibility/cli_accessibility_tester.py --verbose
```

## Contributing

To extend the accessibility system:

1. **Add new violation detectors** in `violation_detector.py`
2. **Extend compliance checking** in `compliance_checker.py`
3. **Add keyboard test cases** in `keyboard_navigator.py`
4. **Customize report formats** in `report_generator.py`

## Support

For issues, questions, or contributions related to accessibility testing:
- Check the [Agent S repository](https://github.com/simular-ai/Agent-S)
- Review existing accessibility documentation
- Submit issues with detailed error messages and screenshots

---

**Making the web accessible for everyone! 🌟**