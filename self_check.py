"""Command-line self-check utility for Agent S dependencies and permissions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from importlib import import_module, metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_STATUS_LABELS: Dict[str, str] = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}


@dataclass
class CheckResult:
    """Outcome of a single self-check item."""

    name: str
    status: str
    message: str
    remedy: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert the result into a serialisable dictionary."""

        return asdict(self)


@dataclass
class DependencySpec:
    """Specification describing how to import and label a dependency."""

    name: str
    import_name: str
    package_name: Optional[str] = None
    optional: bool = False
    purpose: Optional[str] = None


CORE_DEPENDENCIES: Iterable[DependencySpec] = (
    DependencySpec(
        name="pyautogui",
        import_name="pyautogui",
        purpose="Required for screenshot capture and input control",
    ),
    DependencySpec(
        name="Pillow",
        import_name="PIL",
        package_name="Pillow",
        purpose="Image processing backend for screenshots",
    ),
    DependencySpec(
        name="numpy",
        import_name="numpy",
        purpose="Used across perception and planning modules",
    ),
    DependencySpec(
        name="requests",
        import_name="requests",
        purpose="HTTP client for API integrations",
    ),
    DependencySpec(
        name="openai",
        import_name="openai",
        purpose="OpenAI-compatible provider client",
    ),
    DependencySpec(
        name="anthropic",
        import_name="anthropic",
        optional=True,
        purpose="Required when using Anthropic models",
    ),
    DependencySpec(
        name="tiktoken",
        import_name="tiktoken",
        purpose="Token counting utilities",
    ),
    DependencySpec(
        name="paddleocr",
        import_name="paddleocr",
        optional=True,
        purpose="OCR support for perception tasks",
    ),
    DependencySpec(
        name="paddlepaddle",
        import_name="paddle",
        optional=True,
        purpose="Deep learning runtime used by PaddleOCR",
    ),
    DependencySpec(
        name="pytesseract",
        import_name="pytesseract",
        optional=True,
        purpose="Fallback OCR engine",
    ),
    DependencySpec(
        name="google-genai",
        import_name="google.genai",
        package_name="google-genai",
        optional=True,
        purpose="Needed for Google Gemini providers",
    ),
    DependencySpec(
        name="selenium",
        import_name="selenium",
        optional=True,
        purpose="Browser automation helpers",
    ),
)

_API_KEY_VARS: Dict[str, str] = {
    "OPENAI_API_KEY": "Required for OpenAI-compatible providers (OpenAI, SiliconFlow, Fireworks, etc.)",
    "ANTHROPIC_API_KEY": "Needed when running Anthropic models",
    "TOGETHER_API_KEY": "Needed when using Together AI endpoints",
    "SILICONFLOW_API_KEY": "Needed when using SiliconFlow endpoints",
    "GOOGLE_API_KEY": "Needed when using Google Gemini",
    "DEEPSEEK_API_KEY": "Needed when using DeepSeek",
}


def check_python_version() -> CheckResult:
    """Validate that the interpreter version falls within the supported range."""

    major, minor = sys.version_info[:2]
    if 3 <= major:
        if 9 <= minor <= 12 or major > 3:
            return CheckResult(
                name="Python version",
                status="PASS",
                message=f"Detected Python {major}.{minor}",
            )
    return CheckResult(
        name="Python version",
        status="FAIL",
        message=f"Python {major}.{minor} is unsupported. Agent S requires Python 3.9-3.12.",
        remedy="Install Python 3.9-3.12 and recreate your virtual environment.",
    )


def check_dependency(spec: DependencySpec) -> CheckResult:
    """Import a dependency and report whether it is available."""

    package_key = spec.package_name or spec.name
    try:
        module = import_module(spec.import_name)
        version: Optional[str] = None
        try:
            version = getattr(module, "__version__", None)
        except Exception:
            version = None
        if version is None:
            try:
                version = metadata.version(package_key)
            except PackageNotFoundError:
                version = "unknown"
        message = f"Found {package_key} {version}" if version else f"Found {package_key}"
        if spec.purpose:
            message += f" — {spec.purpose}"
        return CheckResult(name=f"Dependency: {spec.name}", status="PASS", message=message)
    except ModuleNotFoundError:
        status = "WARN" if spec.optional else "FAIL"
        message = f"{spec.name} not installed."
        if spec.purpose:
            message += f" {spec.purpose}."
        remedy = f"pip install {package_key}" if package_key else "Install missing dependency."
        return CheckResult(
            name=f"Dependency: {spec.name}",
            status=status,
            message=message,
            remedy=remedy,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        status = "WARN" if spec.optional else "FAIL"
        return CheckResult(
            name=f"Dependency: {spec.name}",
            status=status,
            message=f"Error importing {spec.name}: {exc}",
            remedy=f"Reinstall {package_key} or inspect the stack trace.",
        )


def check_logs_directory(base_dir: Path) -> CheckResult:
    """Verify that the logs directory exists and is writable."""

    logs_dir = base_dir / "logs"
    try:
        logs_dir.mkdir(exist_ok=True)
        test_file = logs_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return CheckResult(
            name="Filesystem permissions",
            status="PASS",
            message=f"logs/ directory is writable at {logs_dir.resolve()}",
        )
    except Exception as exc:
        return CheckResult(
            name="Filesystem permissions",
            status="FAIL",
            message=f"Cannot write to {logs_dir}: {exc}",
            remedy="Ensure the current user can create files in the project directory.",
        )


def check_api_keys() -> List[CheckResult]:
    """Detect configured API keys for model and grounding providers."""

    results: List[CheckResult] = []
    detected = [var for var in _API_KEY_VARS if os.environ.get(var)]
    if detected:
        results.append(
            CheckResult(
                name="Model API keys",
                status="PASS",
                message="Detected: " + ", ".join(detected),
            )
        )
    else:
        results.append(
            CheckResult(
                name="Model API keys",
                status="WARN",
                message="No LLM provider API keys detected in environment.",
                remedy="Export OPENAI_API_KEY, ANTHROPIC_API_KEY, SILICONFLOW_API_KEY, or another provider key before running Agent S.",
            )
        )
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        results.append(
            CheckResult(
                name="HF_TOKEN",
                status="PASS",
                message="Detected Hugging Face token (HF_TOKEN)",
            )
        )
    else:
        results.append(
            CheckResult(
                name="HF_TOKEN",
                status="WARN",
                message="HF_TOKEN not set. Required when hosting grounding models like UI-TARS on Hugging Face.",
                remedy="Export HF_TOKEN with a Hugging Face access token that has Inference Endpoint permissions.",
            )
        )
    return results


def check_macos_permissions() -> List[CheckResult]:
    """Inspect macOS automation and screen recording permissions."""

    results: List[CheckResult] = []
    try:
        import Quartz  # type: ignore

        accessibility_ok: Optional[bool] = None
        try:
            options = {Quartz.kAXTrustedCheckOptionPrompt: False}
            accessibility_ok = Quartz.AXIsProcessTrustedWithOptions(options)
        except AttributeError:
            try:
                accessibility_ok = Quartz.AXIsProcessTrusted()
            except AttributeError:
                accessibility_ok = None
        if accessibility_ok is not None:
            if accessibility_ok:
                results.append(
                    CheckResult(
                        name="macOS Accessibility",
                        status="PASS",
                        message="Automation permission granted (Accessibility).",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="macOS Accessibility",
                        status="FAIL",
                        message="Accessibility permission not granted.",
                        remedy="Open System Settings → Privacy & Security → Accessibility and enable Terminal (or your Python IDE) for automation.",
                    )
                )
        else:
            results.append(
                CheckResult(
                    name="macOS Accessibility",
                    status="WARN",
                    message="Unable to verify Accessibility permission (AXIsProcessTrusted unavailable).",
                    remedy="Manually confirm in System Settings → Privacy & Security → Accessibility.",
                )
            )

        if hasattr(Quartz, "CGPreflightScreenCaptureAccess"):
            screen_ok = Quartz.CGPreflightScreenCaptureAccess()
            if screen_ok:
                results.append(
                    CheckResult(
                        name="macOS Screen Recording",
                        status="PASS",
                        message="Screen recording permission granted.",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name="macOS Screen Recording",
                        status="FAIL",
                        message="Screen recording permission not granted.",
                        remedy="Open System Settings → Privacy & Security → Screen Recording and allow Terminal (or your Python IDE).",
                    )
                )
        else:
            results.append(
                CheckResult(
                    name="macOS Screen Recording",
                    status="WARN",
                    message="Cannot verify screen recording permission on this macOS version.",
                    remedy="Manually check System Settings → Privacy & Security → Screen Recording.",
                )
            )
    except ModuleNotFoundError:
        results.append(
            CheckResult(
                name="macOS Permissions",
                status="WARN",
                message="pyobjc not installed; macOS permission checks skipped.",
                remedy="pip install pyobjc to enable automated permission checks.",
            )
        )
    except Exception as exc:  # pragma: no cover - safety net
        results.append(
            CheckResult(
                name="macOS Permissions",
                status="WARN",
                message=f"Unable to verify macOS permissions: {exc}",
                remedy="Check Accessibility and Screen Recording permissions manually.",
            )
        )
    return results


def check_linux_dependencies() -> List[CheckResult]:
    """Confirm Linux-specific dependencies required by pyautogui."""

    results: List[CheckResult] = []
    if shutil.which("scrot"):
        results.append(
            CheckResult(
                name="Linux screenshot backend",
                status="PASS",
                message="Found 'scrot' command (required by pyautogui).",
            )
        )
    else:
        results.append(
            CheckResult(
                name="Linux screenshot backend",
                status="WARN",
                message="'scrot' not found. pyautogui screenshots may fail on Linux.",
                remedy="Install scrot via your package manager (e.g., sudo apt install scrot).",
            )
        )
    return results


def check_windows_dependencies() -> List[CheckResult]:
    """Ensure Windows automation helpers are present."""

    results: List[CheckResult] = []
    try:
        import_module("pywinauto")
        results.append(
            CheckResult(
                name="pywinauto",
                status="PASS",
                message="pywinauto available for Windows automation.",
            )
        )
    except ModuleNotFoundError:
        results.append(
            CheckResult(
                name="pywinauto",
                status="WARN",
                message="pywinauto not installed. Required for Windows automation features.",
                remedy="pip install pywinauto",
            )
        )
    try:
        import_module("win32api")
        results.append(
            CheckResult(
                name="pywin32",
                status="PASS",
                message="pywin32 available.",
            )
        )
    except ModuleNotFoundError:
        results.append(
            CheckResult(
                name="pywin32",
                status="WARN",
                message="pywin32 not installed. Required for Windows automation features.",
                remedy="pip install pywin32",
            )
        )
    return results


def check_screenshot(skip: bool = False) -> Optional[CheckResult]:
    """Attempt to capture a screenshot, unless the user opted out."""

    if skip:
        return CheckResult(
            name="Screenshot capture",
            status="WARN",
            message="Screenshot test skipped by user request.",
        )
    try:
        import pyautogui

        pyautogui.screenshot()
        return CheckResult(
            name="Screenshot capture",
            status="PASS",
            message="pyautogui.screenshot() succeeded.",
        )
    except Exception as exc:
        return CheckResult(
            name="Screenshot capture",
            status="WARN",
            message=f"Screenshot capture failed: {exc}",
            remedy="Grant screen recording permission and ensure a display is available.",
        )


def collect_checks(skip_screenshot: bool) -> List[CheckResult]:
    """Aggregate all self-check results into a single list."""

    results: List[CheckResult] = []
    results.append(check_python_version())
    results.extend(check_dependency(spec) for spec in CORE_DEPENDENCIES)
    results.append(check_logs_directory(Path.cwd()))
    results.extend(check_api_keys())

    current_os = platform.system()
    if current_os == "Darwin":
        results.extend(check_macos_permissions())
    elif current_os == "Linux":
        results.extend(check_linux_dependencies())
    elif current_os == "Windows":
        results.extend(check_windows_dependencies())

    screenshot_result = check_screenshot(skip_screenshot)
    if screenshot_result:
        results.append(screenshot_result)

    return results


def print_results(results: List[CheckResult]) -> None:
    """Render self-check results in a tabular text format."""

    name_width = max((len(result.name) for result in results), default=10)
    header = f"{'Status':<8} {'Check':<{name_width}} Details"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for result in results:
        status_display = _STATUS_LABELS.get(result.status, result.status)
        print(f"{status_display:<8} {result.name:<{name_width}} {result.message}")
        if result.remedy:
            print(f"         {'':<{name_width}} ↪ {result.remedy}")
    print(separator)

    summary = {
        "pass": sum(result.status == "PASS" for result in results),
        "warn": sum(result.status == "WARN" for result in results),
        "fail": sum(result.status == "FAIL" for result in results),
    }
    print(
        f"Summary → PASS: {summary['pass']} · WARN: {summary['warn']} · FAIL: {summary['fail']}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the Agent S self-check command."""

    parser = argparse.ArgumentParser(
        description="Run Agent S permission and dependency self-checks."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--skip-screenshot",
        action="store_true",
        help="Skip the live screenshot capture test (for headless environments).",
    )
    args = parser.parse_args(argv)

    results = collect_checks(skip_screenshot=args.skip_screenshot)

    if args.json:
        payload = [result.to_dict() for result in results]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_results(results)

    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
