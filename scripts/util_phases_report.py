"""Attribute nvidia-smi samples from slurm/bench_util_phases.sbatch to run phases.

Phases are reconstructed from the job log: the run starts at the "[phase] start"
marker, checkpoint loading runs until the first "[eval]" line, each "[eval] ...
(Ns)" line consumes N seconds of generation/scoring, and the gaps between eval
blocks are the refinement (gradient) sweeps.

Usage: python scripts/util_phases_report.py logs/bench_util_phases-<jobid>.out
"""
import re
import sys
from datetime import datetime, timedelta

log_path = sys.argv[1]
csv_path = log_path.replace(".out", ".csv")

start = end = None
evals = []  # (task, seconds)
for line in open(log_path, errors="replace"):
    m = re.match(r"\[phase\] start (\d\d:\d\d:\d\d)", line)
    if m:
        start = m.group(1)
    m = re.match(r"\[phase\] end (\d\d:\d\d:\d\d)", line)
    if m:
        end = m.group(1)
    m = re.match(r"\[eval\] (\w+): .*\((\d+)s\)", line)
    if m:
        evals.append((m.group(1), int(m.group(2))))

samples = []  # (datetime, util, watts)
for line in open(csv_path, errors="replace"):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4 or "%" not in parts[1]:
        continue
    try:
        ts = datetime.strptime(parts[0].split(".")[0], "%Y/%m/%d %H:%M:%S")
    except ValueError:
        continue
    samples.append((ts, float(parts[1].rstrip(" %")), float(parts[2].rstrip(" W"))))

if not samples or not start:
    sys.exit(f"no usable samples/markers in {log_path}")

day = samples[0][0].date()
t0 = datetime.combine(day, datetime.strptime(start, "%H:%M:%S").time())
t_end = (datetime.combine(day, datetime.strptime(end, "%H:%M:%S").time())
         if end else samples[-1][0])

# The first eval begins once all checkpoints are loaded; find it by walking the
# eval durations backwards from the end of the run is unreliable (refinement in
# between), so mark load = t0 .. first sustained-busy sample.
busy = [s for s in samples if s[1] >= 50 and s[0] >= t0]
t_first_busy = busy[0][0] if busy else t0

total_eval_s = sum(s for _, s in evals)


def stats(sel, label):
    if not sel:
        return
    u = [x[1] for x in sel]
    p = [x[2] for x in sel]
    print(f"  {label:26s} n={len(sel):4d}  util {min(u):3.0f}-{max(u):3.0f}% "
          f"(mean {sum(u)/len(u):3.0f}%)  power {min(p):3.0f}-{max(p):3.0f} W "
          f"(mean {sum(p)/len(p):3.0f} W)")


print(f"run {t0.time()} -> {t_end.time()}  "
      f"({(t_end - t0).total_seconds()/60:.0f} min), {len(samples)} samples")
print(f"evals logged: {len(evals)} totalling {total_eval_s/60:.0f} min "
      f"({', '.join(f'{t}:{s}s' for t, s in evals[:4])}, ...)")
print()
stats([s for s in samples if t0 <= s[0] < t_first_busy], "checkpoint loading")
stats([s for s in samples if s[0] >= t_first_busy], "compute (eval+refine)")
print()
# Split compute samples by utilization mode: generation eval keeps the GPU
# saturated; the coding task's scoring runs generated code on CPU (GPU idle).
comp = [s for s in samples if s[0] >= t_first_busy]
stats([s for s in comp if s[1] >= 90], "  saturated (>=90%)")
stats([s for s in comp if 10 <= s[1] < 90], "  partial (10-90%)")
stats([s for s in comp if s[1] < 10], "  idle (<10%, CPU scoring)")
