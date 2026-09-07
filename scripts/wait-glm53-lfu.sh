#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do
  [ -f "$Q/done/310-glm53-lfu2.sh" ]   && { echo "LFU DONE"; exit 0; }
  [ -f "$Q/failed/310-glm53-lfu2.sh" ] && { echo "LFU FAILED"; exit 0; }
  sleep 60
done
echo TIMEOUT
