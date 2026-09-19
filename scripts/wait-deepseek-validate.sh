#!/usr/bin/env bash
# Block until the checkpoint is complete and independently validated, or until
# something in that chain fails. Polls the durable queue rather than a process,
# so a WSL restart mid-wait does not read as success.
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 480); do   # up to 8h at 60s
  [ -f "$Q/done/110-deepseek-validate-recovery.sh" ]   && { echo "VALIDATED"; exit 0; }
  [ -f "$Q/failed/110-deepseek-validate-recovery.sh" ] && { echo "VALIDATION FAILED"; exit 0; }
  [ -f "$Q/failed/100-deepseek-recovery-resume.sh" ]   && { echo "RECOVERY FAILED"; exit 0; }
  sleep 60
done
echo "TIMEOUT"
