"""Shadow interface evolution framework.

Side-effect on import: set PROXY_CONFIG_FILE to an OSWorld template copy
if the env var isn't already set and one of the templates carries it.

This has to happen BEFORE `desktop_env` is imported anywhere under
agent_s4/, because desktop_env.controllers.setup initializes its
global ProxyPool from PROXY_CONFIG_FILE at module-import time. The
default is a cwd-relative path that only resolves when run from inside
a clone — the orchestrator runs from the repo root, one level up.

Both OSWorld templates ship the same dataimpulse.json, so the first one
present wins; a checkout with only one of them still resolves.
"""

import os

from agent_s4.paths import PROJECT_ROOT as _REPO_ROOT

if "PROXY_CONFIG_FILE" not in os.environ:
    for _template in ("OSWorld_Evol", "OSWorldV2_Evol"):
        _proxy = (
            _REPO_ROOT
            / _template
            / "evaluation_examples"
            / "settings"
            / "proxy"
            / "dataimpulse.json"
        )
        if _proxy.is_file():
            os.environ["PROXY_CONFIG_FILE"] = str(_proxy)
            break
