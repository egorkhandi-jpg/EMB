#!/usr/bin/env bash
# На экземпляре, в tmux: модели по очереди через fast_download.py, метка models/<имя>/.done.
cd "$HOME/embodied-lm-stand" || exit 1
for m in "$@"; do
  /venv/main/bin/python tools_remote/fast_download.py "EleutherAI/$m" --threads 24 >> download_fast.log 2>&1 && touch "models/$m/.done"
done
