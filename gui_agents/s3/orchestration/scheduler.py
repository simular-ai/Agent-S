# gui_agents/s3/orchestration/scheduler.py
"""TaskScheduler — cron jobs e execuções periódicas via APScheduler.

Envolve APScheduler BackgroundScheduler em API estável para o Agent-S3:
add_cron, add_interval, add_one_shot, start, shutdown. Jobs chamam callables
que tipicamente submetem tarefas ao DAG executor ou à API.

Dep: ``pip install apscheduler``. Import guardado — módulo carrega sem a dep,
erro claro só ao instanciar.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

from gui_agents.s3.logging_utils.structured_logger import get_logger

logger = get_logger("desktopenv.agent.scheduler")


def _require_apscheduler():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # noqa: F401
        from apscheduler.triggers.cron import CronTrigger  # noqa: F401
        from apscheduler.triggers.date import DateTrigger  # noqa: F401
        from apscheduler.triggers.interval import IntervalTrigger  # noqa: F401
    except ImportError as exc:  # pragma: no cover — dep gate
        raise ImportError(
            "APScheduler não instalado. Rode: pip install apscheduler"
        ) from exc


class TaskScheduler:
    """Wrapper sobre BackgroundScheduler APScheduler."""

    def __init__(self, timezone: str = "UTC") -> None:
        _require_apscheduler()
        from apscheduler.schedulers.background import BackgroundScheduler

        self._sched = BackgroundScheduler(timezone=timezone)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._sched.start()
            self._started = True
            logger.info("scheduler_started")

    def shutdown(self, wait: bool = True) -> None:
        if self._started:
            self._sched.shutdown(wait=wait)
            self._started = False
            logger.info("scheduler_shutdown")

    def add_cron(
        self,
        func: Callable[..., Any],
        job_id: str,
        *,
        cron: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        replace_existing: bool = True,
    ) -> str:
        """Adiciona job cron. ``cron`` = expressão padrão APScheduler
        (ex: '0 9 * * 1-5' = 9h weekdays)."""
        from apscheduler.triggers.cron import CronTrigger

        trigger = CronTrigger.from_crontab(cron)
        self._sched.add_job(
            func,
            trigger=trigger,
            args=args or (),
            kwargs=kwargs or {},
            id=job_id,
            replace_existing=replace_existing,
        )
        logger.info("scheduler_cron_added", extra={"job_id": job_id, "cron": cron})
        return job_id

    def add_interval(
        self,
        func: Callable[..., Any],
        job_id: str,
        *,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        replace_existing: bool = True,
    ) -> str:
        from apscheduler.triggers.interval import IntervalTrigger

        trigger = IntervalTrigger(
            seconds=seconds, minutes=minutes, hours=hours
        )
        self._sched.add_job(
            func,
            trigger=trigger,
            args=args or (),
            kwargs=kwargs or {},
            id=job_id,
            replace_existing=replace_existing,
        )
        logger.info(
            "scheduler_interval_added",
            extra={"job_id": job_id, "minutes": minutes, "seconds": seconds},
        )
        return job_id

    def add_one_shot(
        self,
        func: Callable[..., Any],
        job_id: str,
        *,
        run_at: datetime,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> str:
        from apscheduler.triggers.date import DateTrigger

        self._sched.add_job(
            func,
            trigger=DateTrigger(run_date=run_at),
            args=args or (),
            kwargs=kwargs or {},
            id=job_id,
            replace_existing=True,
        )
        logger.info(
            "scheduler_oneshot_added",
            extra={"job_id": job_id, "run_at": run_at.isoformat()},
        )
        return job_id

    def remove(self, job_id: str) -> None:
        self._sched.remove_job(job_id)
        logger.info("scheduler_job_removed", extra={"job_id": job_id})

    def list_jobs(self) -> list[dict[str, Any]]:
        return [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
                "trigger": str(j.trigger),
            }
            for j in self._sched.get_jobs()
        ]