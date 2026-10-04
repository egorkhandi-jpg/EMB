#!/usr/bin/env bash
# Состояние экземпляра: хвост лога очереди, nvidia-smi, свободное место, сессии tmux.
# remote_status.sh [очередь] [строк]
q="${1:-}"; n="${2:-15}"
ssh vast "cd ~/embodied-lm-stand; tmux ls 2>/dev/null || echo 'tmux: сессий нет'; \
  if [ -n '$q' ]; then echo '--- $q.log'; tail -n $n '$q.log' 2>/dev/null; echo '--- $q.out'; tail -n 5 '$q.out' 2>/dev/null; fi; \
  echo '---'; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader; \
  df -h / | tail -1; free -g | sed -n 2p" 2>&1 | grep -v -E "Welcome to vast|Have fun|AI agents"
