"""Soak test: sustained concurrent load against a running dlmserve server.

Verifies stability under high concurrency — no crashes, OOM, or 5xx errors.
Reports p50/p99 latency and checks all required Prometheus metrics.

Usage:
    python benchmarks/soak_test.py --url http://localhost:8000 --duration 600 --concurrency 100

Requires the server to be running: dlmserve
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from collections import Counter

import aiohttp

PAYLOAD = {
    "model": "dlmserve",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one sentence."}],
    "max_tokens": 64,
    "num_denoising_steps": 16,
    "temperature": 0.0,
}

REQUIRED_METRICS = [
    "dlmserve_requests_in_flight",
    "dlmserve_denoising_step_duration_seconds",
    "dlmserve_step_batch_size",
    "dlmserve_committed_tokens_per_request",
    "dlmserve_throughput_tokens_per_second",
    "dlmserve_queue_depth",
]


async def _one_request(
    session: aiohttp.ClientSession,
    url: str,
    latencies: list[float],
    status_counts: Counter,
) -> None:
    t0 = time.monotonic()
    try:
        async with session.post(f"{url}/v1/chat/completions", json=PAYLOAD) as resp:
            await resp.read()
            status_counts[resp.status] += 1
            if resp.status == 200:
                latencies.append(time.monotonic() - t0)
    except Exception as exc:
        status_counts[f"error:{type(exc).__name__}"] += 1


async def _check_metrics(session: aiohttp.ClientSession, url: str) -> list[str]:
    async with session.get(f"{url}/metrics") as resp:
        body = await resp.text()
    missing = [m for m in REQUIRED_METRICS if m not in body]
    return missing


async def run(url: str, duration: int, concurrency: int) -> int:
    latencies: list[float] = []
    status_counts: Counter = Counter()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(session: aiohttp.ClientSession) -> None:
        async with sem:
            await _one_request(session, url, latencies, status_counts)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Pre-flight: health check
        try:
            async with session.get(f"{url}/health") as r:
                if r.status != 200:
                    print(f"FAIL: /health returned {r.status}")
                    return 1
        except Exception as exc:
            print(f"FAIL: server not reachable at {url} — {exc}")
            return 1

        missing = await _check_metrics(session, url)

        deadline = time.monotonic() + duration
        tasks: set[asyncio.Task] = set()

        print(f"Soak test: {concurrency} concurrent, {duration}s, target {url}")
        print("Press Ctrl+C to stop early.\n")

        t_report = time.monotonic() + 10
        while time.monotonic() < deadline:
            err_total = sum(v for k, v in status_counts.items() if k != 200)
            if err_total > 50:
                print(f"\n  Server unreachable — stopping early (errors={err_total})")
                break
            if len(tasks) < concurrency:
                t = asyncio.create_task(bounded(session))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
            else:
                await asyncio.sleep(0.01)

            if time.monotonic() >= t_report:
                done = status_counts[200]
                errs = sum(v for k, v in status_counts.items() if k != 200)
                elapsed = duration - (deadline - time.monotonic())
                p50 = statistics.median(latencies) if latencies else 0
                print(f"  t={elapsed:.0f}s  ok={done}  err={errs}  p50={p50:.2f}s")
                t_report += 10

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Final metrics check (server may already be down)
        try:
            missing = await _check_metrics(session, url)
        except Exception:
            missing = ["server unreachable — metrics not checked"]

    # Results
    total = sum(status_counts.values())
    ok = status_counts[200]
    err = total - ok
    print(f"\n{'=' * 60}")
    print(f"Total requests : {total}")
    print(f"OK (200)       : {ok}")
    print(f"Errors         : {err}  {dict(status_counts)}")
    if latencies:
        latencies.sort()
        print(f"p50 latency    : {statistics.median(latencies):.2f}s")
        print(f"p99 latency    : {latencies[int(len(latencies) * 0.99)]:.2f}s")
        print(f"max latency    : {latencies[-1]:.2f}s")

    print("\nMetrics check:")
    if missing:
        for m in missing:
            print(f"  MISSING: {m}")
    else:
        print(f"  All {len(REQUIRED_METRICS)} metrics present in /metrics")

    failed = err > 0 or bool(missing)
    print(f"\n{'PASS' if not failed else 'FAIL'}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=600, help="seconds")
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.url, args.duration, args.concurrency)))


if __name__ == "__main__":
    main()
