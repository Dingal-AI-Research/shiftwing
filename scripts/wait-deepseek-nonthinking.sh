#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 90); do
  [ -f "$Q/done/150-deepseek-nonthinking-probe.sh" ]   && { echo "PROBE DONE"; exit 0; }
  [ -f "$Q/failed/150-deepseek-nonthinking-probe.sh" ] && { echo "PROBE FAILED"; exit 0; }
  sleep 60
done
echo TIMEOUT
