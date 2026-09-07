#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do   # ~8h at 2 min
  [ -f "$Q/done/230-glm53-direct.sh" ]   && { echo "DIRECT RUN DONE"; exit 0; }
  [ -f "$Q/failed/230-glm53-direct.sh" ] && { echo "DIRECT RUN FAILED"; exit 0; }
  sleep 120
done
echo "TIMEOUT"
