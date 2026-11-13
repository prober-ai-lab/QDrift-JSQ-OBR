#!/usr/bin/env bash
# QDrift campaign progress check (x40908).
# Reports: queue sizes + breakdown, running jobs, results count, recent
# completions, GPU snapshot, and the current select_mix selections.
#
# Usage:   scripts/campaign_status.sh [--mix] [--log N]
#   --mix     also run select_mix.py (the (alpha,gamma) selection table)
#   --log N   show the last N campaign-log lines from docs/paper-plan.md (default 0)
#
# Env overrides: SSH_HOST, REMOTE_REPO, REMOTE_QUEUE, REMOTE_PY
set -u

SSH_HOST="${SSH_HOST:-xiaoxiao@x40908.gao.whu.com}"
REMOTE_REPO="${REMOTE_REPO:-/data1/xiaoxiao/QDrift-JSQ-OBR}"
REMOTE_QUEUE="${REMOTE_QUEUE:-/data1/xiaoxiao/qdrift_v2}"
REMOTE_PY="${REMOTE_PY:-/data1/xiaoxiao/reef/.venv/bin/python}"
SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes)

WANT_MIX=0
LOG_N=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mix) WANT_MIX=1; shift ;;
    --log) LOG_N="${2:-20}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Run a remote command with retries (x40908 SSH drops connections often).
rssh() {
  local i
  for i in 1 2 3 4; do
    if timeout 60 ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$@"; then
      return 0
    fi
    echo "  (ssh retry $i ...)" >&2
    sleep 3
  done
  echo "!! SSH to $SSH_HOST failed after 4 tries" >&2
  return 1
}

echo "========================================================"
echo " QDrift campaign status  —  $(date '+%Y-%m-%d %H:%M %Z')"
echo " host: $SSH_HOST"
echo "========================================================"

rssh REMOTE_QUEUE="$REMOTE_QUEUE" REMOTE_REPO="$REMOTE_REPO" bash -s <<'REMOTE'
set -u
qf="$REMOTE_QUEUE/queue_fast.txt"
qs="$REMOTE_QUEUE/queue.txt"

echo
echo "### FAST queue (quick W4A16/W2A16, GPUs 0,6) : $( [ -f "$qf" ] && wc -l < "$qf" || echo '?') jobs left"
[ -f "$qf" ] && cut -f1 "$qf" | sed -E 's/_a[0-9.]+$//; s/_obr$//; s/_score$//; s/_obs$//' | sort | uniq -c | sed 's/^/    /'

echo
echo "### SLOW queue (heavy W4A4/pairwise/8B/13B, GPUs 1,2,3) : $( [ -f "$qs" ] && wc -l < "$qs" || echo '?') jobs left"
[ -f "$qs" ] && cut -f1 "$qs" | sed -E 's/_a[0-9.]+$//; s/_obr$//; s/_score$//; s/_obs$//' | sort | uniq -c | sed 's/^/    /'

echo
echo "### Runners alive"
pgrep -af 'run_queue(_fast)?\.sh' | grep -v 'pgrep' | sed -E 's/^([0-9]+).*(run_queue[_a-z]*\.sh [0-9,]+).*/    pid \1  \2/' | sort -u || echo "    (none — RUNNERS DOWN!)"

echo
echo "### Running compression jobs (model / sparsity / objective)"
n=$(pgrep -cf 'python main.py run')
echo "    $n job(s) running:"
for pid in $(pgrep -f 'python main.py run'); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  m=$(echo "$cmd"   | grep -oE -- '--model [^ ]+'    | awk '{print $2}')
  s=$(echo "$cmd"   | grep -oE -- '--sparsity [0-9.]+' | awk '{print $2}')
  wb=$(echo "$cmd"  | grep -oE -- '--w-bits [0-9]+'  | awk '{print $2}')
  ab=$(echo "$cmd"  | grep -oE -- '--a-bits [0-9]+'  | awk '{print $2}')
  c=$(echo "$cmd"   | grep -oE -- '--comp [a-z]+'    | awk '{print $2}')
  mx=$(echo "$cmd"  | grep -oE -- '--mix-asym [0-9.]+' | awk '{print $2}')
  off=$(echo "$cmd" | grep -oE -- '--offload')
  et=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  echo "    pid $pid  ${m:-?}  w${wb:-?}a${ab:-?} s${s:-?}  ${c:-?}${mx:+ a$mx} ${off:+[offload]}  (up $et)"
done

echo
echo "### Results on disk : $(ls "$REMOTE_REPO/results/v2" 2>/dev/null | wc -l)"
echo "### 5 most recent completions:"
for f in $(ls -t "$REMOTE_REPO/results/v2"/*.json 2>/dev/null | head -5); do
  t=$(stat -c '%y' "$f" 2>/dev/null | cut -c1-16)
  echo "    $t  $(basename "$f")"
done

echo
echo "### GPU snapshot (physical GPU6 is dead — queried ids skip it)"
nvidia-smi -i 0,1,2,3,4,5,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
  | sed 's/^/    gpu /' || echo "    (nvidia-smi unavailable)"
REMOTE

if [ "$WANT_MIX" -eq 1 ]; then
  echo
  echo "### select_mix selections (by val PPL; ties -> OBR)"
  rssh "cd '$REMOTE_REPO' && PYTHONPATH='$REMOTE_REPO' '$REMOTE_PY' visualize/tables/select_mix.py results/v2 2>&1"
fi

if [ "$LOG_N" -gt 0 ]; then
  PLAN="$(dirname "$0")/../docs/paper-plan.md"
  if [ -f "$PLAN" ]; then
    echo
    echo "### Last $LOG_N lines of docs/paper-plan.md (local)"
    tail -n "$LOG_N" "$PLAN" | sed 's/^/    /'
  fi
fi

echo
echo "done."
