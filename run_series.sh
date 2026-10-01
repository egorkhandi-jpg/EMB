#!/usr/bin/env bash
# Серия: муха, линейный контроль, предобученная LM, та же архитектура со случайными весами
# (обе LM со стандартизацией). Случайная сеть сеется от посева (stand_gpu.py, с 2026-10-01).
# Использование: bash run_series.sh <модель> <число посевов> [пауза, с] [доп. флаги stand_gpu.py...]
# Пример: bash run_series.sh EleutherAI/pythia-70m 10 10
#         OUT=results_qwen bash run_series.sh Qwen/Qwen2.5-1.5B 10 10 --dtype bfloat16 --chunk 128
# Посев, у которого уже есть result_*.json, пропускается — серию можно продолжить после остановки.
MODEL=${1:-EleutherAI/pythia-70m}
SEEDS=${2:-10}
PAUSE=${3:-10}   # пауза между LM-прогонами; 10 с — решение Егора 2026-10-01 (было 60), остывание стережёт cool()
shift 3 2>/dev/null
EXTRA="$@"
OUT=${OUT:-results}
TAG=$(basename "$MODEL")
mkdir -p "$OUT"

temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }
# Правило 5 CLAUDE.md: выше 85 °C — ждать остывания ниже 70 °C.
cool() {
  local t; t=$(temp); echo "$(date +%T) GPU ${t} °C" | tee -a "$OUT/thermal.log"
  if [ -n "$t" ] && [ "$t" -gt 85 ]; then
    while t=$(temp); [ "$t" -ge 70 ]; do echo "$(date +%T) остывание: ${t} °C" | tee -a "$OUT/thermal.log"; sleep 30; done
  fi
}
# RAN=1, только если прогон действительно считался: пауза после пропущенного посева
# не нужна (ИСПРАВЛЕНО 2026-10-01 — перезапуск с паузой 60 с терял бы ~16 мин на sleep).
run() {  # run <имя результата> <флаги...>
  local name=$1; shift; RAN=0
  if [ -f "$OUT/result_${name}.json" ]; then echo "есть: $name"; return; fi
  RAN=1; cool; python stand_gpu.py "$@" --out "$OUT" || echo "ОШИБКА: $name" | tee -a "$OUT/errors.log"
}
for s in $(seq 1 $SEEDS); do
  run "fly_s$s" --cond fly --seed $s
  run "lin_s$s" --cond lin --seed $s
  run "${TAG}_pretrained_c_s$s" --cond lm --model $MODEL --weights pretrained --seed $s $EXTRA; [ $RAN = 1 ] && sleep $PAUSE
  run "${TAG}_random_c_s$s" --cond lm --model $MODEL --weights random --seed $s $EXTRA; [ $RAN = 1 ] && sleep $PAUSE
done
python summarize.py "$OUT"
