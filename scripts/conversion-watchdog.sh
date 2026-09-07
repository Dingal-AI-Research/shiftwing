#!/usr/bin/env bash
# Keep a long Hub conversion near line rate.
#
# Two failure modes, both observed on this job today:
#
#   1. Silent stall -- the process stays alive on a dead socket and the log
#      freezes. Cost five hours before anyone noticed. The supervisor's retry
#      loop only fires when python exits, which a hung socket never does.
#   2. Throughput decay -- the connection degrades under sustained use. Measured
#      46 MiB/s in the morning and 11 by evening; killing the converter and
#      letting the supervisor respawn it restored 34 MiB/s at once, so the decay
#      is in the connection rather than the route.
#
# Both get the same treatment: kill the converter and let the supervisor's loop
# bring it back. Conversion is resumable by hash, so a restart costs only the
# partial shard in flight.
#
# Progress is judged from the staging directory, which --delete-source shrinks
# as shards are consumed; negative deltas are clamped so a deletion is never
# mistaken for a stall.
set -uo pipefail
ROOT=/home/dinga/Projects/shiftwing
OUT="$ROOT/c/qwen38fn_int8"
STAGING="$ROOT/c/.qwen38fn.source"
LOG="$ROOT/.nightqueue/watchdog.log"

STALL_S=${STALL_S:-900}        # no movement at all for this long
SLOW_MIBS=${SLOW_MIBS:-12}     # sustained rate under this counts as decayed
SLOW_STRIKES=${SLOW_STRIKES:-3}
COOLDOWN_S=${COOLDOWN_S:-600}  # never restart more often than this
SAMPLE_S=${SAMPLE_S:-120}

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"; }
shards() { ls "$OUT"/*.safetensors 2>/dev/null | wc -l; }
# Progress signals. The staging directory is deliberately churned by
# --delete-source, so NO aggregate byte measure over it is meaningful: net `du`
# growth is ~0 at full line rate, and summing .incomplete files goes negative
# when one completes. Both were tried and both were wrong.
#
# What is reliable is (a) the committed shard count and (b) the size of the one
# largest in-flight download. Either advancing means the job is alive; a rate is
# only computed when the same file is still being written between samples.
biggest() {
  find "$STAGING" -name '*.incomplete' -printf '%s %p\n' 2>/dev/null \
    | sort -rn | head -1
}

converter() {
  local p c
  for p in /proc/[0-9]*/cmdline; do
    c=$(tr '\0' ' ' < "$p" 2>/dev/null) || continue
    case "$c" in
      *convert_qwen.py' '--repo*) basename "$(dirname "$p")"; return ;;
    esac
  done
}

bounce() {
  log "restarting converter $1: $2"
  kill "$1" 2>/dev/null
  sleep 8
  kill -9 "$1" 2>/dev/null
  find "$STAGING" -name '*.incomplete' -size 0 -delete 2>/dev/null
  last_bounce=$(date +%s)
  strikes=0
}

log "watchdog v2 up (stall ${STALL_S}s | slow <${SLOW_MIBS}MiB/s x${SLOW_STRIKES} | cooldown ${COOLDOWN_S}s)"
prev_s=$(shards); prev_line=$(biggest); prev_f=${prev_line#* }; prev_z=${prev_line%% *}
last_move=$(date +%s); last_bounce=0; strikes=0

while true; do
  sleep "$SAMPLE_S"
  pid=$(converter)
  if [ -z "$pid" ]; then
    if ls "$ROOT"/.nightqueue/running/090-* > /dev/null 2>&1; then
      last_move=$(date +%s); continue
    fi
    log "job no longer running; watchdog exiting"; exit 0
  fi

  s_now=$(shards); line=$(biggest); f=${line#* }; z=${line%% *}; now=$(date +%s)
  moved=0; rate=-1
  [ "$s_now" != "$prev_s" ] && moved=1                     # a shard committed
  if [ -n "$f" ] && [ "$f" = "$prev_f" ] && [ "${z:-0}" -gt "${prev_z:-0}" ]; then
    moved=1                                               # same file still growing
    rate=$(( (z - prev_z) / 1048576 / SAMPLE_S ))
  elif [ -n "$f" ] && [ "$f" != "$prev_f" ]; then
    moved=1                                               # moved on to a new file
  fi
  [ "$moved" -eq 1 ] && last_move=$now
  prev_s=$s_now; prev_f=$f; prev_z=$z

  if [ $(( now - last_move )) -ge "$STALL_S" ] && [ $(( now - last_bounce )) -ge "$COOLDOWN_S" ]; then
    bounce "$pid" "stalled $(( now - last_move ))s: shards flat and no download growth"
    continue
  fi

  # Only a measured same-file rate counts toward the decay strikes.
  if [ "$rate" -ge 0 ] && [ "$rate" -lt "$SLOW_MIBS" ]; then
    strikes=$(( strikes + 1 ))
    log "slow ${rate} MiB/s (strike ${strikes}/${SLOW_STRIKES}, shards ${s_now})"
    if [ "$strikes" -ge "$SLOW_STRIKES" ] && [ $(( now - last_bounce )) -ge "$COOLDOWN_S" ]; then
      bounce "$pid" "throughput decayed to ${rate} MiB/s"
    fi
  elif [ "$rate" -ge "$SLOW_MIBS" ]; then
    strikes=0
  fi
done
