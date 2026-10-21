"""Project-wide constants.

Moved out of island_common.py during the framework refactor. These
describe experiment geometry (cache paths, eval set layout) rather than
per-experiment configuration — ExperimentConfig fields take precedence
when they overlap.
"""

from __future__ import annotations

# SQLite file (under repo root) that holds the run's evolution state.
# All modules that open the DB (driver, orchestrator, db) reference this
# name so the path is defined in one place.
DB_FILENAME = "shadow_evolution.db"

# Subdirectory (under repo root) that holds eval task-list JSON files
# (eval_island_*.json + eval_heldout_*.json). Centralized here so callers
# never hardcode the directory name.
EVAL_SETS_DIR = "eval_sets"

# Historical note: there used to be an EVAL_SETS dict mapping tier-level
# → eval_island_*.json filename, plus a TIER_ORDER constant used by
# operators' promotion cascade. Both were retired in v2 — variants now
# run on the single canonical train_set_file (see
# ExperimentConfig.train_set_file). The `variants.eval_level` DB column
# is retained for legacy-row compatibility but is not read or written by
# v2 code.

# Directory (under repo root) where per-variant eval trajectories are cached.
EVAL_CACHE_DIR = "eval_cache"

# Directory (under repo root) where per-mutation builder logs are written.
BUILDER_LOGS_DIR = "builder_logs"

# Directory (under repo root) where migration phase 1 stages outbound
# variants for phase 2 destinations to pick up. Each source island writes
# to MIGRATE_TRANSIT_DIR/<source_island>/ via tmp+rename for atomicity.
# The presence of MIGRATE_TRANSIT_DIR/<src>/ marks "phase 1 done for src"
# during resume; phase 2 reads and then deletes the dir.
MIGRATE_TRANSIT_DIR = "_migrate_transit"

# Note: the promote-threshold (fraction of tier-N tasks a variant must
# pass to earn tier N+1 eval) was retired in v2 along with the tier
# cascade. Variants now evaluate on the single canonical train_set_file.
