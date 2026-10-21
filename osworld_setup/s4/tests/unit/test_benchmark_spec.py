from agent_s4.benchmark_spec import BENCHMARKS, ns


def test_known_benchmarks_present():
    assert set(BENCHMARKS) == {"osworld", "scienceboard", "osworld_v2"}
    for b in BENCHMARKS.values():
        assert set(b) == {"template", "base", "tag"}


def test_osworld_keeps_bare_names():
    assert ns("shadow_evolution.db", "osworld") == "shadow_evolution.db"
    assert ns("eval_cache", "osworld") == "eval_cache"
    assert ns("_migrate_transit", "osworld") == "_migrate_transit"


def test_scienceboard_namespaces_dirs_and_files():
    assert ns("eval_cache", "scienceboard") == "eval_cache_scienceboard"
    assert ns("builder_logs", "scienceboard") == "builder_logs_scienceboard"
    assert ns("_migrate_transit", "scienceboard") == "_migrate_transit_scienceboard"
    assert ns("shadow_evolution.db", "scienceboard") == "shadow_evolution_scienceboard.db"


def test_template_and_base_are_distinct():
    assert BENCHMARKS["osworld"]["template"] == "OSWorld_Evol"
    assert BENCHMARKS["scienceboard"]["template"] == "ScienceBoard_Evol"
    assert BENCHMARKS["scienceboard"]["base"] == "ScienceBoard_Base"


def test_osworld_v2_namespaces_off_osworld():
    """osworld_v2 must never share workspace state with the v1 osworld runs
    the paper draws on — separate DB, cache, logs, clones."""
    assert BENCHMARKS["osworld_v2"]["template"] == "OSWorldV2_Evol"
    assert BENCHMARKS["osworld_v2"]["base"] == "OSWorldV2_Base"
    assert ns("shadow_evolution.db", "osworld_v2") == "shadow_evolution_osworld_v2.db"
    assert ns("eval_cache", "osworld_v2") == "eval_cache_osworld_v2"
    assert ns("builder_logs", "osworld_v2") == "builder_logs_osworld_v2"
    assert ns("_migrate_transit", "osworld_v2") == "_migrate_transit_osworld_v2"


def test_builder_logs_and_transit_namespaced():
    assert ns("builder_logs", "scienceboard") == "builder_logs_scienceboard"
    assert ns("_migrate_transit", "scienceboard") == "_migrate_transit_scienceboard"
    assert ns("builder_logs", "osworld") == "builder_logs"
    assert ns("_migrate_transit", "osworld") == "_migrate_transit"


def test_optional_run_namespace_is_appended(monkeypatch):
    monkeypatch.setenv("SHADOW_EVOL_RUN_NAMESPACE", "prime25v15")
    assert ns("shadow_evolution.db", "osworld_v2") == (
        "shadow_evolution_osworld_v2_prime25v15.db"
    )
    assert ns("eval_cache", "osworld_v2") == (
        "eval_cache_osworld_v2_prime25v15"
    )


def test_run_namespace_rejects_path_syntax(monkeypatch):
    import pytest

    monkeypatch.setenv("SHADOW_EVOL_RUN_NAMESPACE", "../other-run")
    with pytest.raises(ValueError, match="SHADOW_EVOL_RUN_NAMESPACE"):
        ns("shadow_evolution.db", "osworld_v2")
