#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do   # ~8h at 2 min
  [ -f "$Q/done/220-glm53-remeasure.sh" ]   && { echo "REMEASURE DONE"; exit 0; }
  [ -f "$Q/failed/220-glm53-remeasure.sh" ] && { echo "REMEASURE FAILED"; exit 0; }
  sleep 120
done
echo "TIMEOUT"
