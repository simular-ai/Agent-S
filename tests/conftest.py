"""Shared pytest fixtures for gui-agents test suite."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test file operations.

    Yields:
        Path: Path to a temporary directory that is cleaned up after the test.

    Example:
        def test_file_creation(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("hello")
            assert test_file.exists()
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Create a temporary file for test operations.

    Yields:
        Path: Path to a temporary file that is cleaned up after the test.

    Example:
        def test_file_reading(temp_file):
            temp_file.write_text("test content")
            assert temp_file.read_text() == "test content"
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    yield tmp_path

    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def mock_env_vars(monkeypatch) -> Dict[str, str]:
    """Provide a fixture for setting temporary environment variables.

    Args:
        monkeypatch: pytest's monkeypatch fixture for modifying environment.

    Returns:
        Dict[str, str]: Dictionary to store environment variable overrides.

    Example:
        def test_env_var(mock_env_vars):
            mock_env_vars["TEST_VAR"] = "test_value"
            assert os.getenv("TEST_VAR") == "test_value"
    """
    env_vars = {}

    def set_env(key: str, value: str) -> None:
        env_vars[key] = value
        monkeypatch.setenv(key, value)

    # Return a dict-like object that sets env vars on assignment
    class EnvVarSetter(dict):
        def __setitem__(self, key: str, value: str) -> None:
            super().__setitem__(key, value)
            set_env(key, value)

    return EnvVarSetter(env_vars)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary for testing.

    Returns:
        Dict[str, Any]: Sample configuration with common test settings.

    Example:
        def test_config_loading(sample_config):
            assert sample_config["timeout"] == 30
            assert sample_config["verbose"] is True
    """
    return {
        "timeout": 30,
        "verbose": True,
        "max_retries": 3,
        "api_key": "test_api_key_12345",
        "endpoints": {
            "primary": "https://api.example.com",
            "backup": "https://backup.example.com",
        },
    }


@pytest.fixture
def mock_api_response() -> Dict[str, Any]:
    """Provide a mock API response for testing API interactions.

    Returns:
        Dict[str, Any]: Mock API response with typical structure.

    Example:
        def test_api_parsing(mock_api_response):
            assert mock_api_response["status"] == "success"
            assert "data" in mock_api_response
    """
    return {
        "status": "success",
        "code": 200,
        "data": {
            "id": "test_id_123",
            "result": "test_result",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        "message": "Operation completed successfully",
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment state before and after each test.

    This fixture runs automatically for every test and ensures a clean
    environment by preserving and restoring the original working directory.

    Example:
        This fixture is autouse=True, so it runs automatically:

        def test_changing_directory(temp_dir):
            os.chdir(temp_dir)
            # Directory is automatically reset after test
    """
    original_dir = os.getcwd()
    yield
    os.chdir(original_dir)


@pytest.fixture
def captured_logs(caplog):
    """Provide access to captured log messages during tests.

    Args:
        caplog: pytest's caplog fixture for capturing log output.

    Returns:
        The caplog fixture configured for easy log testing.

    Example:
        def test_logging(captured_logs):
            logger.info("test message")
            assert "test message" in captured_logs.text
    """
    return caplog
