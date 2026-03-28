#!/usr/bin/env bash
set -euo pipefail

URL="${EVAL_URL:-http://localhost:8766}"
AGENT_DIR="${1:-agent}"

if [ ! -f "$AGENT_DIR/controller.py" ]; then
  echo "No controller.py found in $AGENT_DIR" >&2
  exit 1
fi

# Zip agent dir
ZIPFILE=$(mktemp /tmp/agent-XXXX.zip)
rm -f "$ZIPFILE"
trap 'rm -f "$ZIPFILE"' EXIT
(cd "$AGENT_DIR" && zip -qr "$ZIPFILE" . -x '__pycache__/*')

# Name from content hash
HASH=$(sha256sum "$ZIPFILE" | cut -c1-8)
NAME="agent_${HASH}"
echo "Submitting as $NAME"

# Submit
RESP=$(curl -s -X POST "$URL/submit" \
  -H "X-Agent-Name: $NAME" \
  -H "Content-Type: application/zip" \
  --data-binary "@$ZIPFILE")

JOB_ID=$(echo "$RESP" | jq -r '.job_id // empty')
if [ -z "$JOB_ID" ]; then
  echo "Submit failed: $RESP" >&2
  exit 1
fi
echo "Job $JOB_ID"

# Poll
while true; do
  sleep 2
  RESULT=$(curl -s "$URL/status/$JOB_ID")
  STATUS=$(echo "$RESULT" | jq -r '.status')

  case "$STATUS" in
    queued)   echo "Queued (position $(echo "$RESULT" | jq -r '.position'))..." ;;
    running)  echo "Running..." ;;
    converged|ok)
      RATING=$(echo "$RESULT" | jq -r '.rating')
      CI=$(echo "$RESULT" | jq -r '.ci')
      GAMES=$(echo "$RESULT" | jq -r '.games')
      WINS=$(echo "$RESULT" | jq -r '.wins')
      LOSSES=$(echo "$RESULT" | jq -r '.losses')
      TIES=$(echo "$RESULT" | jq -r '.ties')
      SAVED_NAME=$(echo "$RESULT" | jq -r '.name')
      echo "Rating: $RATING ± $CI ($GAMES games, ${WINS}W-${LOSSES}L-${TIES}T)"
      echo "Saved as: $SAVED_NAME"
      echo "---"
      echo "rating:           $RATING"
      echo "ci:               $CI"
      echo "wins:             $WINS"
      echo "losses:           $LOSSES"
      echo "ties:             $TIES"
      echo "games:            $GAMES"
      exit 0 ;;
    error)
      echo "$RESULT" | jq -r '"Error: \(.reason)\nRating at abort: \(.rating) ± \(.ci)"'
      exit 1 ;;
    *)
      echo "Unknown: $RESULT" >&2
      exit 1 ;;
  esac
done
