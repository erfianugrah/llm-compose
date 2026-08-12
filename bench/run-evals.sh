#!/usr/bin/env bash
# bench/run-evals.sh - thin entrypoint wrapper for run-evals.py
exec python3 /work/run-evals.py "$@"
