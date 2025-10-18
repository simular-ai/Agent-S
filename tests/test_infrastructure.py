"""Validation tests to verify testing infrastructure setup."""

import sys
from pathlib import Path

import pytest


def test_python_version():
    """Verify Python version is within the supported range."""
    version = sys.version_info
    assert version.major == 3
    assert 9 <= version.minor <= 12, f"Python version {version.minor} not in supported range (3.9-3.12)"


def test_imports_available():
    """Verify all critical testing dependencies are importable."""
    import pytest
    import pytest_cov
    import pytest_mock

    assert pytest is not None
    assert pytest_cov is not None
    assert pytest_mock is not None


def test_project_structure():
    """Verify the project structure is correct."""
    project_root = Path(__file__).parent.parent
    assert (project_root / "gui_agents").exists(), "gui_agents package directory not found"
    assert (project_root / "tests").exists(), "tests directory not found"
    assert (project_root / "tests" / "unit").exists(), "tests/unit directory not found"
    assert (project_root / "tests" / "integration").exists(), "tests/integration directory not found"
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml not found"


def test_conftest_fixtures(temp_dir, temp_file, sample_config, mock_api_response):
    """Verify shared fixtures from conftest.py work correctly."""
    # Test temp_dir fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")
    assert test_file.read_text() == "hello"

    # Test temp_file fixture
    assert temp_file.exists()
    temp_file.write_text("test content")
    assert temp_file.read_text() == "test content"

    # Test sample_config fixture
    assert isinstance(sample_config, dict)
    assert "timeout" in sample_config
    assert sample_config["timeout"] == 30

    # Test mock_api_response fixture
    assert isinstance(mock_api_response, dict)
    assert mock_api_response["status"] == "success"
    assert "data" in mock_api_response


def test_env_vars_fixture(mock_env_vars):
    """Verify environment variables fixture works correctly."""
    import os

    mock_env_vars["TEST_KEY"] = "test_value"
    assert os.getenv("TEST_KEY") == "test_value"


@pytest.mark.unit
def test_unit_marker():
    """Verify unit test marker is registered."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Verify integration test marker is registered."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Verify slow test marker is registered."""
    assert True


def test_pytest_mock_plugin(mocker):
    """Verify pytest-mock plugin is working."""
    mock_func = mocker.Mock(return_value=42)
    result = mock_func()
    assert result == 42
    mock_func.assert_called_once()


class TestClassExample:
    """Verify class-based tests are discovered."""

    def test_class_based_test(self):
        """Verify class-based test methods work."""
        assert True

    def test_multiple_assertions(self):
        """Verify multiple assertions in a single test."""
        assert 1 + 1 == 2
        assert "hello".upper() == "HELLO"
        assert [1, 2, 3] == [1, 2, 3]
