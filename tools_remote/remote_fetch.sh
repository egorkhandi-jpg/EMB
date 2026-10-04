#!/usr/bin/env bash
# Забрать результаты с экземпляра и сделать бэкап (правило 6): remote_fetch.sh <папка> [очередь]
# Внешнего хранилища нет: удаление экземпляра уничтожает всё незабранное.
set -euo pipefail
cd "$(dirname "$0")/.."
d="$1"; q="${2:-}"
scp -q -r "vast:~/embodied-lm-stand/$d" ./ 2>&1 | grep -v -E "Welcome to vast|Have fun|AI agents" || true
if [ -n "$q" ]; then
  for f in "$q.log" "$q.out" "$q.err" "$q.done" "$q.failed"; do
    scp -q "vast:~/embodied-lm-stand/$f" "./$f.vast" 2>/dev/null || true
  done
fi
b="backup/${d%/}_$(date +%Y-%m-%d)"
[ -e "$b" ] && b="${b}_$(date +%H%M)"
cp -r "$d" "$b"
echo "забрано: $d → ./$d, бэкап $b"
echo "$(date '+%Y-%m-%d %H:%M:%S') забрано $d${q:+ ($q)}" >> tools_remote/runs.log
