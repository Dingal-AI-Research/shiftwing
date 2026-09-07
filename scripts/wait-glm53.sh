#!/usr/bin/env bash
# Block until the GLM lane is either validated or definitively broken.
# Polls the durable queue rather than any process, so a WSL restart mid-wait
# does not turn into a false "finished".
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 330); do   # ~11h at 2 min
  if [ -f "$Q/done/210-glm53-validate.sh" ]; then echo "VALIDATED"; exit 0; fi
  if [ -f "$Q/failed/210-glm53-validate.sh" ]; then echo "VALIDATION FAILED"; exit 0; fi
  if [ -f "$Q/failed/200-glm53-convert.sh" ]; then echo "CONVERSION FAILED"; exit 0; fi
  sleep 120
done
echo "TIMEOUT after ~11h"
exit 0
