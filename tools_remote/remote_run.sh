#!/usr/bin/env bash
# Запуск очереди на экземпляре в tmux (сессия — имя очереди без расширения).
# remote_run.sh <очередь> [аргументы run_queue.py, напр. --break-min 0]
# Файл очереди копируется на экземпляр (это не код); код — только через remote_push.sh.
# Время старта пишется в tools_remote/runs.log (локально) и в <очередь>.log на экземпляре.
set -euo pipefail
cd "$(dirname "$0")/.."
q="$1"; shift || true
[ -f "$q" ] || { echo "нет файла очереди $q" >&2; exit 1; }
name=$(basename "$q" .txt | tr '.' '_')
scp -q "$q" "vast:~/embodied-lm-stand/$q" 2>&1 | grep -v -E "Welcome to vast|Have fun|AI agents" || true
ssh vast "cd ~/embodied-lm-stand && git log --oneline -1 && \
  echo \"\$(date '+%Y-%m-%d %H:%M:%S') удалённый старт очереди (tmux $name, \$(git rev-parse --short HEAD))\" >> $q.log && \
  tmux new-session -d -s $name \"export PATH=/venv/main/bin:\\$PATH; python run_queue.py $q $* > $q.out 2> $q.err\" && tmux ls" \
  2>&1 | grep -v -E "Welcome to vast|Have fun|AI agents"
echo "$(date '+%Y-%m-%d %H:%M:%S') старт $q на vast (tmux $name) $*" >> tools_remote/runs.log
