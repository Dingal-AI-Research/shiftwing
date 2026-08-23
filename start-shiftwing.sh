#!/bin/bash
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
exec ./c/shiftwing serve --model ./c/ornith397 --host 127.0.0.1 --port 8000 --context 16384 --max-tokens 1024 --cuda --ram-gb 18 --cuda-expert-gb 5 --kv-slots 1 --model-id ornith-397b
