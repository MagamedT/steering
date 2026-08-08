from __future__ import annotations

import asyncio


async def run_ranked_jobs(jobs, worker_count, run_one):
    """Keep at most one job active per logical actor rank."""
    next_index = 0
    in_flight = {}

    def start(rank, job):
        task = asyncio.create_task(run_one(rank, *job))
        in_flight[task] = rank

    for rank in range(min(worker_count, len(jobs))):
        start(rank, jobs[next_index])
        next_index += 1

    while in_flight:
        completed, _ = await asyncio.wait(
            in_flight, return_when=asyncio.FIRST_COMPLETED
        )
        for task in completed:
            rank = in_flight.pop(task)
            try:
                result = await task
            except Exception as error:
                result = error
            yield rank, result

            if next_index < len(jobs):
                start(rank, jobs[next_index])
                next_index += 1
