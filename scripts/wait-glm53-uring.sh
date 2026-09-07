#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do
  [ -f "$Q/done/250-glm53-uring.sh" ]   && { echo "URING RUN DONE"; exit 0; }
  [ -f "$Q/failed/250-glm53-uring.sh" ] && { echo "URING RUN FAILED"; exit 0; }
  sleep 120
done
echo "TIMEOUT"
