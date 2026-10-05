#!/usr/bin/env bash
# Скачивание моделей на экземпляре (в tmux): повторы при обрыве, лог в ~/embodied-lm-stand/download.log.
# download_models.sh <модель> [<модель> ...]
cd "$HOME/embodied-lm-stand" || exit 1
for m in "$@"; do
  for try in 1 2 3 4 5 6; do
    if timeout 1800 /venv/main/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$m', allow_patterns=['*.json', '*.safetensors', '*.txt'])
" >> download.log 2>&1; then echo "$(date '+%F %T') $m готово (попытка $try)" >> download.log; break; fi
    echo "$(date '+%F %T') $m обрыв, попытка $try" >> download.log; sleep 10
  done
done
echo "$(date '+%F %T') скачивание завершено" >> download.log
