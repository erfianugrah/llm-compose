#!/usr/bin/env bash
# init-webui.sh — one-shot Open WebUI configuration
#
# Imports workspace model configs (system prompts, parameters, capabilities)
# from webui/models.json via the Open WebUI API.
#
# Authentication (checked in order):
#   1. WEBUI_API_KEY env var
#   2. WEBUI_ADMIN_EMAIL + WEBUI_ADMIN_PASSWORD env vars (signs in to get token)
#   3. Interactive prompt for email/password
#
# Idempotent — safe to re-run. Updates existing models, creates new ones.
#
# Requires: curl, jq

set -euo pipefail

WEBUI_URL="${WEBUI_URL:-http://localhost:3000}"
MODELS_FILE="${MODELS_FILE:-$(dirname "$0")/../webui/models.json}"

# ── Helpers ──────────────────────────────────────────────────────────

die() { echo "Error: $*" >&2; exit 1; }

check_deps() {
    command -v curl >/dev/null || die "curl is required"
    command -v jq >/dev/null || die "jq is required (apt install jq)"
}

wait_for_health() {
    echo "Waiting for Open WebUI at ${WEBUI_URL}..."
    local attempts=0
    while ! curl -sf "${WEBUI_URL}/health" >/dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 60 ]; then
            die "Open WebUI not reachable after 60 attempts. Is 'make up' running?"
        fi
        sleep 2
    done
    echo "Open WebUI is healthy."
}

# ── Authentication ───────────────────────────────────────────────────

get_token() {
    # Method 1: API key from env
    if [ -n "${WEBUI_API_KEY:-}" ]; then
        echo "$WEBUI_API_KEY"
        return
    fi

    local email="${WEBUI_ADMIN_EMAIL:-}"
    local password="${WEBUI_ADMIN_PASSWORD:-}"

    # Method 2: email/password from env
    # Method 3: interactive prompt
    if [ -z "$email" ] || [ -z "$password" ]; then
        echo ""
        echo "No WEBUI_API_KEY or WEBUI_ADMIN_EMAIL/PASSWORD found."
        echo "Enter your Open WebUI admin credentials:"
        read -rp "  Email: " email
        read -rsp "  Password: " password
        echo ""
    fi

    [ -n "$email" ] || die "Email is required"
    [ -n "$password" ] || die "Password is required"

    # Sign in
    local resp
    resp=$(curl -sf "${WEBUI_URL}/api/v1/auths/signin" \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"${email}\",\"password\":\"${password}\"}" 2>&1) \
        || die "Sign-in failed. Check credentials. Response: ${resp}"

    local token
    token=$(echo "$resp" | jq -r '.token // empty')
    [ -n "$token" ] || die "No token in sign-in response: ${resp}"

    echo "$token"
}

# ── Model import ─────────────────────────────────────────────────────

import_models() {
    local token="$1"

    [ -f "$MODELS_FILE" ] || die "Models file not found: ${MODELS_FILE}"

    echo "Importing workspace models from ${MODELS_FILE}..."

    local models
    models=$(jq -c '.' "$MODELS_FILE")

    local resp
    resp=$(curl -sf "${WEBUI_URL}/api/v1/models/import" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${token}" \
        -d "{\"models\":${models}}" 2>&1) \
        || die "Model import failed. Response: ${resp}"

    # Show results
    local count
    count=$(echo "$resp" | jq -r 'if type == "array" then length elif .models then (.models | length) else 0 end' 2>/dev/null || echo "?")
    echo "Imported ${count} workspace models."

    # List them
    echo ""
    echo "Workspace models:"
    jq -r '.[] | "  \(.name) (\(.base_model_id))"' "$MODELS_FILE"
}

# ── Main ─────────────────────────────────────────────────────────────

main() {
    check_deps
    wait_for_health

    local token
    token=$(get_token)

    import_models "$token"

    echo ""
    echo "Done. Open ${WEBUI_URL} to use the configured models."
}

main "$@"
