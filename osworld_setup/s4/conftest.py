"""Session-wide safety gates, loaded before test-module collection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The package is agent_s4, importable from the Agent-S repository root.
REPO_ROOT = ROOT.parents[1]
os.environ["SHADOW_EVOL_TEST_DENY_NETWORK"] = "1"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9"
os.environ["ALL_PROXY"] = "http://127.0.0.1:9"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

pythonpath = os.environ.get("PYTHONPATH", "")
paths = [str(ROOT), str(REPO_ROOT),
         *(part for part in pythonpath.split(os.pathsep) if part)]
os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
for _path in (REPO_ROOT, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from shadow_evol_test_network_guard import install_network_guard

install_network_guard()


import pytest


# agent_s4.eval.osworld calls load_dotenv() at import time, so importing it
# anywhere in the session pours the developer's real .env into os.environ and
# every later test inherits it. Model-routing variables decide which endpoint a
# built command points at, so a stray one in .env silently rewrites assertions
# in tests that never mentioned it. Clear them per test; a test that wants one
# sets it with monkeypatch, which applies after this fixture.
_ROUTING_ENV = ("SHADOW_EVOL_VM_OPENAI_BASE_URL",)


@pytest.fixture(autouse=True)
def _neutralise_dotenv_routing(monkeypatch):
    for name in _ROUTING_ENV:
        monkeypatch.delenv(name, raising=False)
