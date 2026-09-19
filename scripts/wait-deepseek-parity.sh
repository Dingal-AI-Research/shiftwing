#!/usr/bin/env bash
# Block until complete-model logit parity lands, or until anything before it
# fails. Polls the durable queue, so a WSL restart mid-wait cannot read as
# success -- three restarts have already happened during this work.
Q=/home/dinga/Projects/shiftwing/.nightqueue
for i in $(seq 1 720); do   # up to 12h at 60s
  for j in 110-deepseek-validate-recovery 120-deepseek-prepare-reviews 130-deepseek-full-logit-parity; do
    [ -f "$Q/failed/$j.sh" ] && { echo "FAILED: $j"; exit 0; }
  done
  [ -f "$Q/done/130-deepseek-full-logit-parity.sh" ] && { echo "PARITY DONE"; exit 0; }
  sleep 60
done
echo "TIMEOUT"
