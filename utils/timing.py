import time
from contextlib import contextmanager

# Collects {stage_name: seconds} for the most recent pipeline run.
# Cleared at the start of each run_pipeline() call.
stage_timings: dict = {}


@contextmanager
def timed_stage(name: str):
    start = time.time()
    print(f"⏱️  {name} — starting...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        stage_timings[name] = elapsed
        print(f"⏱️  {name} — done in {elapsed:.2f}s")


def reset_timings():
    stage_timings.clear()


def print_timing_summary():
    if not stage_timings:
        print("No timings recorded.")
        return
    total = sum(stage_timings.values())
    print("\n" + "=" * 40)
    print("⏱️  PIPELINE TIMING SUMMARY")
    print("=" * 40)
    for name, seconds in stage_timings.items():
        pct = (seconds / total * 100) if total else 0
        print(f"  {name:<20} {seconds:>7.2f}s  ({pct:5.1f}%)")
    print("-" * 40)
    print(f"  {'TOTAL':<20} {total:>7.2f}s")
    print("=" * 40 + "\n")