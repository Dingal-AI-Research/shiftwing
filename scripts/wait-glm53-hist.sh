#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do
  [ -f "$Q/done/290-glm53-expert-hist.sh" ]   && { echo "HIST DONE"; exit 0; }
  [ -f "$Q/failed/290-glm53-expert-hist.sh" ] && { echo "HIST FAILED"; exit 0; }
  sleep 60
done
echo TIMEOUT
