"""Harness interfaces + result dataclasses.

BuilderHarness and EvaluatorHarness are intentionally separate — they model
two different launch shapes (one subprocess per mutation vs. 25 parallel
VMs running an agent class) and have no shared method signatures worth
abstracting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BuilderResult:
    exit_code: int
    cost_usd: float
    log_path: Path
    crashed: bool  # auth / rate-limit / short-exit signal


@dataclass
class EvalResults:
    pass_rate: float
    total: int
    passed: int
    failed: int
    results: list[dict] = field(default_factory=list)
    timed_out: bool = False


class BuilderHarness(ABC):
    @abstractmethod
    def launch(
        self,
        *,
        clone_dir: Path,
        prompt: str,
        model: str,
        timeout: int | None,
        env_extra: dict[str, str] | None,
        variant_id: str,
        logs_dir: Path,
        log_name: str,  # e.g. "gui-if-007_mutation" or "gui-if-008_crossover"
        builder_bin: str | None = None,  # explicit claude binary path; None → PATH
    ) -> BuilderResult: ...


class EvaluatorHarness(ABC):
    @abstractmethod
    def run_tasks(
        self,
        *,
        clone_dir: Path,
        shadow_dir: Path,
        result_dir: Path,
        task_ids: list[str],
        model: str,
        region: str,
        timeout: int,
        max_steps: int | None = None,
        no_time_limit: bool = False,
        harness: str | None = None,
        benchmark: str = "osworld",  # selects the *_Base task tree
    ) -> EvalResults: ...
