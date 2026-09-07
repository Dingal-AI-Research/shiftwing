#!/usr/bin/env bash
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 240); do
  [ -f "$Q/done/270-glm53-mtp-accept.sh" ]   && { echo "MTP ACCEPTANCE DONE"; exit 0; }
  [ -f "$Q/failed/270-glm53-mtp-accept.sh" ] && { echo "MTP ACCEPTANCE FAILED"; exit 0; }
  sleep 120
done
echo "TIMEOUT"
