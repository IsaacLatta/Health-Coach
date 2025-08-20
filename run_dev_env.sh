#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  . "$ROOT/.env"
  set +a
else
  echo "ERROR: .env not found in $ROOT (continuing with shell env)"
fi

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_HOST="${BACKEND_HOST%\"}"; BACKEND_HOST="${BACKEND_HOST#\"}"
BACKEND_PORT="${BACKEND_PORT:-8080}"

FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_HOST="${FRONTEND_HOST%\"}"; FRONTEND_HOST="${FRONTEND_HOST#\"}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"

export BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"

FRONTEND_FILE="${FRONTEND_FILE:-}"
if [[ -z "$FRONTEND_FILE" ]]; then
  for cand in \
    "$ROOT/src/health_coach/frontend/home_page.py"
  do
    [[ -f "$cand" ]] && FRONTEND_FILE="$cand" && break
  done
fi
[[ -z "$FRONTEND_FILE" ]] && { echo "ERROR: Could not find Streamlit entry. Set FRONTEND_FILE=…"; exit 1; }

pids=()
cleanup() {
  echo -e "\nINFO: shutting down..."
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait "${pids[@]:-}" 2>/dev/null || true
  echo -e "DONE."
}
trap cleanup INT TERM

echo "INFO: Frontend running on: http://${FRONTEND_HOST}:${FRONTEND_PORT} (BACKEND_URL=${BACKEND_URL})"
streamlit run "$FRONTEND_FILE" \
  --server.port "$FRONTEND_PORT" \
  --server.address "$FRONTEND_HOST" \
  --server.headless true &

pids+=($!)

echo "INFO: Backend running on: http://${BACKEND_HOST}:${BACKEND_PORT}"
python - <<PY
import os
from health_coach.backend.entry import app
host = os.getenv("BACKEND_HOST","127.0.0.1").strip('"')
port = int(os.getenv("BACKEND_PORT","8000"))
app.run(host=host, port=port, debug=True, use_reloader=False) # drop reloader for trap
PY
