#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do
  [ -f "$Q/done/240-glm53-cache.sh" ]   && { echo "CACHE RUN DONE"; exit 0; }
  [ -f "$Q/failed/240-glm53-cache.sh" ] && { echo "CACHE RUN FAILED"; exit 0; }
  sleep 120
done
echo "TIMEOUT"
