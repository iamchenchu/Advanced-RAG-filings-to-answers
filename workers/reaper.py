"""
workers/reaper.py

Jobs stuck in 'processing' past the timeout mean a worker died mid-job. The
reaper resets them to 'pending' (idempotent pipeline => safe retry), or to
'dead' once attempts are exhausted — dead-lettered for a human, never retried
forever (retry loops on a poisoned document would eat the whole worker pool).

Run:  python workers/reaper.py --once      # one sweep (cron-friendly)
      python workers/reaper.py             # sweep every 5 minutes
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings                        # noqa: E402
from app.persistence import repository                     # noqa: E402
from app.persistence.db import close_pool, run_migrations  # noqa: E402


async def sweep(once: bool) -> None:
    s = get_settings()
    await run_migrations()
    while True:
        n = await repository.reap_stuck_jobs(
            s.ingestion_stuck_job_timeout_min, s.ingestion_max_attempts)
        print(f"reaper: reset {n} stuck job(s)")
        if once:
            return
        await asyncio.sleep(300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    async def run() -> None:
        try:
            await sweep(args.once)
        finally:
            await close_pool()      # same event loop as the pool — clean exit

    asyncio.run(run())


if __name__ == "__main__":
    main()
