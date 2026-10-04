#!/usr/bin/env bash
# Выполняется НА ЭКЗЕМПЛЯРЕ внутри tmux (вызывает remote_run.sh): окружение /venv/main, раннер-очередь, лог старта/конца.
# tmux_queue.sh <очередь> [аргументы run_queue.py]
cd "$HOME/embodied-lm-stand" || exit 1
export PATH=/venv/main/bin:$PATH HF_HUB_DISABLE_SYMLINKS_WARNING=1
q="$1"; shift
echo "$(date '+%Y-%m-%d %H:%M:%S') удалённый старт очереди (tmux, $(git rev-parse --short HEAD), $(python --version 2>&1))" >> "$q.log"
python run_queue.py "$q" "$@" > "$q.out" 2> "$q.err"
echo "$(date '+%Y-%m-%d %H:%M:%S') удалённый конец очереди (rc=$?)" >> "$q.log"
