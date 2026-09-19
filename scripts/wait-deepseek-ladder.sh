#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 480); do
  [ -f "$Q/done/160-deepseek-full-ladder-capability.sh" ]   && { echo "LADDER MEASURED"; exit 0; }
  [ -f "$Q/failed/160-deepseek-full-ladder-capability.sh" ] && { echo "LADDER RUN FAILED"; exit 0; }
  sleep 60
done
echo TIMEOUT
