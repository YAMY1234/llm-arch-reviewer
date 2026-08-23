#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

VIEWER_HOST="${VIEWER_HOST:-127.0.0.1}"
VIEWER_PORT="${VIEWER_PORT:-8766}"
USER_ID="$(id -u)"
TEMP_ROOT="${TMPDIR:-/tmp}"
STATE_DIR="${VIEWER_STATE_DIR:-${TEMP_ROOT%/}/llm-arch-reviewer-viewer-${USER_ID}}"
PID_FILE="$STATE_DIR/viewer-${VIEWER_PORT}.pid"
LOG_FILE="$STATE_DIR/viewer-${VIEWER_PORT}.log"
BROKER_LOG_FILE="$STATE_DIR/viewer-${VIEWER_PORT}-launcher.log"
HEALTH_URL="http://${VIEWER_HOST}:${VIEWER_PORT}/viewer.html"

mkdir -p "$STATE_DIR"

launcher_mode() {
  if [[ -n "${VIEWER_LAUNCH_MODE:-}" ]]; then
    printf '%s\n' "$VIEWER_LAUNCH_MODE"
  elif [[ "$(uname -s)" == "Darwin" \
    && -n "${CODEX_SESSION_ID:-}" \
    && -x "$(command -v osascript 2>/dev/null || true)" ]]; then
    # Codex cleans up descendants of an exec command even when they use nohup.
    # Let Terminal own the nohup launch so the server is outside that process tree.
    printf '%s\n' "terminal-nohup"
  else
    printf '%s\n' "nohup"
  fi
}

read_pid() {
  if [[ -f "$PID_FILE" ]]; then
    tr -d '[:space:]' < "$PID_FILE"
  fi
}

is_running() {
  local pid command
  pid="$(read_pid)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$command" == *"scripts/serve_viewer.py"* && "$command" == *"--port ${VIEWER_PORT}"* ]]
}

listener_description() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$VIEWER_PORT" -sTCP:LISTEN 2>/dev/null || true
  fi
}

start_nohup() {
  local python_bin pid
  python_bin="$(command -v python3)"
  nohup "$python_bin" "$REPO_ROOT/scripts/serve_viewer.py" \
    --host "$VIEWER_HOST" --port "$VIEWER_PORT" "$@" \
    >> "$LOG_FILE" 2>&1 </dev/null &
  pid="$!"
  printf '%s\n' "$pid" > "$PID_FILE"
  disown "$pid" 2>/dev/null || true
}

start_through_terminal() {
  local repo_q broker_log_q inner_command shell_command
  local -a inner_args
  inner_args=(
    env
    "VIEWER_LAUNCH_MODE=nohup"
    "VIEWER_STATE_DIR=$STATE_DIR"
    "VIEWER_HOST=$VIEWER_HOST"
    "VIEWER_PORT=$VIEWER_PORT"
    "$SCRIPT_PATH"
    start
    "$@"
  )
  printf -v repo_q '%q' "$REPO_ROOT"
  printf -v broker_log_q '%q' "$BROKER_LOG_FILE"
  printf -v inner_command '%q ' "${inner_args[@]}"
  shell_command="cd ${repo_q} && ${inner_command}>> ${broker_log_q} 2>&1; exit"

  osascript - "$shell_command" >/dev/null <<'APPLESCRIPT'
on run argv
  tell application "Terminal"
    do script (item 1 of argv)
  end tell
end run
APPLESCRIPT
}

start_server() {
  if is_running; then
    local pid
    pid="$(read_pid)"
    echo "viewer already running: pid $pid"
    echo "$HEALTH_URL"
    return 0
  fi

  rm -f "$PID_FILE"
  local listener
  listener="$(listener_description)"
  if [[ -n "$listener" ]]; then
    echo "port $VIEWER_PORT is already in use:" >&2
    echo "$listener" >&2
    return 1
  fi

  : >> "$LOG_FILE"
  case "$(launcher_mode)" in
    terminal-nohup)
      start_through_terminal "$@"
      ;;
    nohup)
      start_nohup "$@"
      ;;
    *)
      echo "unsupported VIEWER_LAUNCH_MODE: $(launcher_mode)" >&2
      return 2
      ;;
  esac

  local attempt pid
  for attempt in {1..100}; do
    pid="$(read_pid)"
    if curl -fsS --max-time 1 -o /dev/null "$HEALTH_URL" 2>/dev/null; then
      echo "viewer started: pid ${pid:-unknown} · $(launcher_mode)"
      echo "$HEALTH_URL"
      echo "log: $LOG_FILE"
      return 0
    fi
    sleep 0.1
  done

  echo "viewer failed to become ready; recent logs:" >&2
  tail -n 20 "$BROKER_LOG_FILE" >&2 2>/dev/null || true
  tail -n 30 "$LOG_FILE" >&2 || true
  pid="$(read_pid)"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  return 1
}

stop_server() {
  local pid
  if ! is_running; then
    rm -f "$PID_FILE"
    echo "viewer is not running on port $VIEWER_PORT"
    return 0
  fi

  pid="$(read_pid)"
  kill "$pid"
  local attempt
  for attempt in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "viewer stopped: pid $pid"
      return 0
    fi
    sleep 0.1
  done
  echo "viewer did not stop after SIGTERM: pid $pid" >&2
  return 1
}

status_server() {
  if is_running; then
    local pid health
    pid="$(read_pid)"
    health="unreachable"
    if curl -fsS --max-time 2 -o /dev/null "$HEALTH_URL" 2>/dev/null; then
      health="HTTP ready"
    fi
    echo "viewer running: pid $pid · detached · $health"
    echo "$HEALTH_URL"
    echo "log: $LOG_FILE"
    [[ "$health" == "HTTP ready" ]]
    return
  fi
  echo "viewer is not running on port $VIEWER_PORT"
  local listener
  listener="$(listener_description)"
  if [[ -n "$listener" ]]; then
    echo "another process is listening on the port:"
    echo "$listener"
  fi
  return 1
}

action="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi

case "$action" in
  start)
    start_server "$@"
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server "$@"
    ;;
  status)
    status_server
    ;;
  logs)
    if [[ -f "$LOG_FILE" ]]; then
      tail -n "${1:-80}" "$LOG_FILE"
    else
      echo "no log yet: $LOG_FILE"
    fi
    ;;
  *)
    echo "usage: $0 [start|stop|restart|status|logs] [serve_viewer.py args...]" >&2
    echo "environment: VIEWER_HOST, VIEWER_PORT, VIEWER_STATE_DIR, VIEWER_LAUNCH_MODE" >&2
    exit 2
    ;;
esac
