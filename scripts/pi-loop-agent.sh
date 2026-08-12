#!/bin/bash
# Loop agent wrapper: pin the llm-compose GPU model for the iteration's
# duration so a concurrent /mode swap (another session) can't abort the
# stream mid-iteration. Unlock on exit. The lock is polite: anyone can
# force past it with POST /mode {"lock": false}.
# Retries the lock for up to 3 min to ride out proxy restarts/swaps.
MODEL=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--model" ]; then MODEL="$a"; fi
  prev="$a"
done

LOCK_PRESET=""
case "$MODEL" in
  llama-server/loop)        LOCK_PRESET="loop-gemma-4-26B-A4B-it-Q4_K_M" ;;
  llama-server/qwen3-coder) LOCK_PRESET="qwen3-coder-30b-a3b-instruct-q4_k_m" ;;
esac

unlock() {
  curl -s -X POST localhost:11434/mode -H 'Content-Type: application/json' \
    -d '{"lock": false}' >/dev/null 2>&1
}

if [ -n "$LOCK_PRESET" ]; then
  LOCKED=0
  for i in $(seq 1 36); do
    RESP=$(curl -s --max-time 10 -X POST localhost:11434/mode \
      -H 'Content-Type: application/json' -d "{\"lock\": \"$LOCK_PRESET\"}" 2>/dev/null)
    if printf '%s' "$RESP" | grep -q '"locked"'; then
      LOCKED=1
      break
    fi
    sleep 5
  done
  if [ "$LOCKED" != "1" ]; then
    echo "wrapper: could not lock $LOCK_PRESET after 3 min" >&2
    exit 1
  fi
  echo "wrapper: locked $LOCK_PRESET" >&2
  trap unlock EXIT
  trap 'exit 1' TERM INT
fi

pi --no-extensions --no-skills "$@"
