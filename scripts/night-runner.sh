#!/usr/bin/env bash
# Durable job runner for long shiftwing work.
#
# WSL2 on this host reboots without warning (30 GB cap on a 32 GB machine, and
# the owner deliberately left .wslconfig alone), which has already killed one
# 12-hour conversion mid-flight. Agent sessions die with it. This runner is a
# systemd *user* service with linger enabled, so it comes back on its own after
# a reboot and picks up where the queue left off.
#
# Queue protocol: drop an executable script in .nightqueue/pending. Jobs run in
# lexical order, one at a time. A job that exits 0 moves to done/, anything else
# to failed/. Output goes to .nightqueue/logs/<job>.log.
#
# Jobs must be idempotent: a reboot mid-job re-runs it from the start.
set -uo pipefail
ROOT=/home/dinga/Projects/shiftwing
Q="$ROOT/.nightqueue"
cd "$ROOT" || exit 1

log(){ printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$Q/runner.log"; }

log "runner started (boot $(uptime -s))"

# A job interrupted by a reboot is left in running/; put it back at the front.
for f in "$Q"/running/*; do
  [ -e "$f" ] || continue
  log "requeueing interrupted job $(basename "$f")"
  mv "$f" "$Q/pending/" 2>/dev/null
done

while true; do
  job=$(ls -1 "$Q"/pending 2>/dev/null | head -1)
  if [ -z "$job" ]; then sleep 30; continue; fi
  mv "$Q/pending/$job" "$Q/running/$job" 2>/dev/null || continue
  log "start $job"
  chmod +x "$Q/running/$job" 2>/dev/null
  "$Q/running/$job" >> "$Q/logs/$job.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then mv "$Q/running/$job" "$Q/done/$job"; log "done $job"
  else mv "$Q/running/$job" "$Q/failed/$job"; log "FAILED $job rc=$rc"; fi
done
