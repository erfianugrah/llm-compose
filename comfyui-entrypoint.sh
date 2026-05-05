#!/bin/sh
# Copy built-in nodes to the volume-mounted custom_nodes/ on startup
# if they don't already exist there. Handles the empty-volume case
# while allowing user modifications to persist.
for src in /app/ComfyUI/custom_nodes_builtin/*/; do
  name=$(basename "$src")
  if [ ! -d "/app/ComfyUI/custom_nodes/$name" ]; then
    echo "[init] Installing built-in node: $name"
    cp -r "$src" "/app/ComfyUI/custom_nodes/$name"
  fi
done
exec python main.py "$@"
