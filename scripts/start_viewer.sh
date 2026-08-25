#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
viewer_host="${VIEWER_HOST:-127.0.0.1}"
viewer_port="${VIEWER_PORT:-8766}"
viewer_build="2026-08-25-selection-sync-v2"
viewer_url="http://${viewer_host}:${viewer_port}/viewer.html?model=qwen40_v2"
log_path="${TMPDIR:-/tmp}/llm-arch-reviewer-viewer-${viewer_port}.log"
tmux_session="llm-arch-reviewer-viewer-${viewer_port}"
python_bin="$(command -v python3)"

if [[ ! "$viewer_port" =~ ^[0-9]+$ ]]; then
  echo "VIEWER_PORT must be numeric: ${viewer_port}" >&2
  exit 1
fi

if ! grep -Fq \
  "<meta name=\"llm-arch-reviewer-viewer-build\" content=\"${viewer_build}\"" \
  "$repo_dir/docs/viewer.html"; then
  echo "Refusing to start: docs/viewer.html is not canonical build ${viewer_build}." >&2
  exit 1
fi

serves_canonical_viewer() {
  curl --fail --silent --show-error --max-time 2 "$viewer_url" \
    | grep -F \
      "<meta name=\"llm-arch-reviewer-viewer-build\" content=\"${viewer_build}\"" \
      >/dev/null
}

if serves_canonical_viewer 2>/dev/null; then
  echo "Viewer ${viewer_build} is already running at ${viewer_url}"
  exit 0
fi

existing_pids="$(lsof -nP -t -iTCP:"${viewer_port}" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$existing_pids" ]]; then
  while IFS= read -r existing_pid; do
    [[ -n "$existing_pid" ]] || continue
    existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
    if [[ "$existing_command" != *"scripts/serve_viewer.py"* ]]; then
      echo "Port ${viewer_port} is owned by another process: ${existing_command}" >&2
      exit 1
    fi
    kill "$existing_pid"
  done <<< "$existing_pids"
fi

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$tmux_session" >/dev/null 2>&1 || true
  printf -v viewer_command \
    'exec %q %q --host %q --port %q >>%q 2>&1' \
    "$python_bin" "$repo_dir/scripts/serve_viewer.py" \
    "$viewer_host" "$viewer_port" "$log_path"
  tmux new-session -d -s "$tmux_session" -c "$repo_dir" "$viewer_command"
else
  nohup "$python_bin" "$repo_dir/scripts/serve_viewer.py" \
    --host "$viewer_host" --port "$viewer_port" \
    </dev/null >"$log_path" 2>&1 &
fi

for _attempt in {1..120}; do
  if serves_canonical_viewer 2>/dev/null; then
    echo "Started viewer ${viewer_build}: ${viewer_url}"
    echo "Log: ${log_path}"
    exit 0
  fi
  sleep 0.25
done

echo "Viewer did not become ready; inspect ${log_path}." >&2
exit 1
