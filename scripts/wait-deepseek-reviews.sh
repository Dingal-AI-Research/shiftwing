#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 480); do
  [ -f "$Q/done/140-deepseek-reviews.sh" ]   && { echo "ACCEPTANCE COMPLETED"; exit 0; }
  [ -f "$Q/failed/140-deepseek-reviews.sh" ] && { echo "ACCEPTANCE STOPPED AT A FAILING CASE"; exit 0; }
  sleep 60
done
echo TIMEOUT
