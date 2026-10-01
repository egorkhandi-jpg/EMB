#!/usr/bin/env bash
# Серия этапа 2 (пререгистрация v2, docs/RESULTS_2026-10-01_flip.md): flip-2, разворот знака.
# Условия: lm pretrained, lm random, lm_shuf pretrained, flywin, linwin при N = 2, 8, 32; gru.
# Использование: bash run_flip.sh <первый посев> <последний посев> [пауза после LM, с]
#   Части по правилу 5 CLAUDE.md: bash run_flip.sh 1 5; перерыв >= 30 мин; bash run_flip.sh 6 10
#   или всё сразу с перерывом и бэкапами: bash run_flip.sh all
# Досчитанные прогоны (есть result_*.json) пропускаются без паузы — серию можно продолжить.
# lm_shuf при N = 2 совпадает с lm: в окне один прошлый такт, переставлять нечего (оставлено
# ради полноты плана, ~45 с на посев).
OUT=${OUT:-results_flip2}
ENVF="--env flip-2 --reversal"
SUF="_rev_flip2"
MODEL=EleutherAI/pythia-70m
TAG=$(basename "$MODEL")
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1

if [ "$1" = all ]; then
  bash "$0" 1 5 "${2:-10}" && cp -r "$OUT" "backup/${OUT}_part1_$(date +%F)" && echo "$(date +%T) бэкап части 1"
  echo "$(date +%T) перерыв 30 мин (правило 5)"; sleep 1800
  bash "$0" 6 10 "${2:-10}" && cp -r "$OUT" "backup/${OUT}_$(date +%F)" && echo "$(date +%T) бэкап серии"
  exit
fi
FROM=${1:-1}; TO=${2:-10}; PAUSE=${3:-10}
mkdir -p "$OUT"

temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }
cool() {
  local t; t=$(temp); echo "$(date +%T) GPU ${t} °C" >> "$OUT/thermal.log"
  if [ -n "$t" ] && [ "$t" -gt 85 ]; then
    while t=$(temp); [ "$t" -ge 70 ]; do echo "$(date +%T) остывание: ${t} °C" | tee -a "$OUT/thermal.log"; sleep 30; done
  fi
}
run() {  # run <имя результата> <флаги...>; RAN=1, только если прогон считался
  local name=$1; shift; RAN=0
  if [ -f "$OUT/result_${name}.json" ]; then echo "есть: $name"; return; fi
  RAN=1; cool
  python stand_flip.py "$@" $ENVF --out "$OUT" | tail -1 || echo "ОШИБКА: $name" | tee -a "$OUT/errors.log"
  [ -f "$OUT/result_${name}.json" ] || echo "НЕТ РЕЗУЛЬТАТА: $name" | tee -a "$OUT/errors.log"
}
for s in $(seq $FROM $TO); do
  run "gru_H56${SUF}_s$s" --cond gru --seed $s
  for N in 2 8 32; do
    run "flywin_N${N}${SUF}_s$s" --cond flywin --N $N --seed $s
    run "linwin_N${N}${SUF}_s$s" --cond linwin --N $N --seed $s
  done
  for N in 2 8 32; do
    run "${TAG}_pretrained_N${N}${SUF}_s$s" --cond lm --weights pretrained --N $N --seed $s; [ $RAN = 1 ] && sleep $PAUSE
    run "${TAG}_random_N${N}${SUF}_s$s" --cond lm --weights random --N $N --seed $s; [ $RAN = 1 ] && sleep $PAUSE
    run "${TAG}_pretrained_shuf_N${N}${SUF}_s$s" --cond lm_shuf --weights pretrained --N $N --seed $s; [ $RAN = 1 ] && sleep $PAUSE
  done
done
echo "$(date +%T) посевы $FROM-$TO готовы: $(ls "$OUT"/result_*.json 2>/dev/null | wc -l) результатов всего"
