#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do
  [ -f "$Q/done/260-glm53-uring2.sh" ]   && { echo "URING2 DONE"; exit 0; }
  [ -f "$Q/failed/260-glm53-uring2.sh" ] && { echo "URING2 FAILED"; exit 0; }
  sleep 120
done
echo "TIMEOUT"
