#!/usr/bin/env bash
# Серия: муха, предобученная LM, та же архитектура со случайными весами (обе со стандартизацией).
# Использование: bash run_series.sh <модель> <число посевов> [пауза между прогонами, с]
# Пример: bash run_series.sh EleutherAI/pythia-70m 10 60
MODEL=${1:-EleutherAI/pythia-70m}
SEEDS=${2:-10}
PAUSE=${3:-60}   # пауза для остывания ноутбука
OUT=results
for s in $(seq 1 $SEEDS); do
  python stand_gpu.py --cond fly --seed $s --out $OUT
  python stand_gpu.py --cond lm --model $MODEL --weights pretrained --seed $s --out $OUT; sleep $PAUSE
  python stand_gpu.py --cond lm --model $MODEL --weights random --seed $s --out $OUT; sleep $PAUSE
done
python summarize.py $OUT
