#!/usr/bin/env bash
set -euo pipefail

collect_payload() {
    local payload=""
    local var value

    for var in \
        CLAUDE_TOOL_NAME \
        CLAUDE_TOOL_INPUT \
        CLAUDE_TOOL_ARGS \
        CLAUDE_TOOL_COMMAND \
        CLAUDE_BASH_COMMAND
    do
        value="${!var-}"
        if [[ -n "${value}" ]]; then
            payload+=$'\n'"${value}"
        fi
    done

    if [[ ! -t 0 ]]; then
        payload+=$'\n'"$(cat 2>/dev/null || true)"
    fi

    printf '%s' "${payload}"
}

block() {
    local message="$1"
    echo "[safety-check] ${message}" >&2
    exit 2
}

payload="$(collect_payload)"
if [[ -z "${payload//[$'\t\r\n ']}" ]]; then
    exit 0
fi

if printf '%s' "${payload}" | grep -Eqi 'git[[:space:]]+reset[[:space:]]+--hard|git[[:space:]]+checkout[[:space:]]+--|rm[[:space:]]+-rf'; then
    block "Blocked destructive shell command. Use an explicit edit tool and get user confirmation first."
fi

if printf '%s' "${payload}" | grep -Eqi '(>|>>|tee|sed[[:space:]]+-i|perl[[:space:]]+-i|mv[[:space:]]|cp[[:space:]]).*(config[\\/][^[:space:]]*_custom\.yaml|src[\\/]models[\\/][^[:space:]]*\.(hef|onnx)|checkpoints[\\/]|calib_data[\\/]|calib_images[\\/])'; then
    block "Blocked a shell write touching protected local-override or artifact paths."
fi

exit 0
