import asyncio

async def run_ranked_jobs(jobs, worker_count, run_one):
    next_idx = 0
    in_flight = {}

    def start(rank, job):
        if len(job) == 3:
            m, slug, label = job
            print(f"→ [gpu {rank}] start model='{m}' concept='{label}' (slug={slug})", flush=True)
        else:
            print(f"→ [gpu {rank}] started: '{job[0]}'", flush=True)
        task = asyncio.create_task(run_one(rank, *job))
        in_flight[task] = rank

    for rank in range(min(worker_count, len(jobs))):
        start(rank, jobs[next_idx])
        next_idx += 1

    while in_flight:
        done, _ = await asyncio.wait(
            in_flight,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            rank = in_flight.pop(task)
            try:
                result = await task
            except Exception as e:
                result = e
            yield rank, result

            if next_idx < len(jobs):
                start(rank, jobs[next_idx])
                next_idx += 1
