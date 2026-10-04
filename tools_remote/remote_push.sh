#!/usr/bin/env bash
# Код на экземпляр Vast: git push в голый репозиторий ~/stand.git и git pull в рабочей копии.
# Код правится только локально и коммитится; на экземпляре не редактировать (CLAUDE.md, «Удалённая машина»).
# Не вызывать, пока на экземпляре идёт очередь: git pull меняет код под идущей серией (правило 9).
# Если pull отказывает из-за неотслеживаемых файлов (забранные результаты потом закоммичены локально) — сверить их с
# закоммиченными (cmp) и отодвинуть в ~/embodied-lm-stand/.pull_aside/, затем повторить (так сделано 2026-10-04).
set -euo pipefail
cd "$(dirname "$0")/.."
branch=$(git branch --show-current)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ВНИМАНИЕ: есть незакоммиченные изменения — на экземпляр уйдёт только закоммиченное" >&2
fi
git push -q vast "$branch" 2>&1 | grep -v -E "Welcome to vast|Have fun|AI agents" || true
ssh vast "cd ~/embodied-lm-stand && git pull -q --ff-only origin $branch && git log --oneline -1" 2>&1 | grep -v -E "Welcome to vast|Have fun|AI agents"
