#!/usr/bin/env bash
# Шаг 1 (шкала): части не длиннее ~2 ч с перерывами (правило 5), бэкап после каждой части.
# Замер времени прохода 4096 позиций, fp32: 160M 140 мс, 410M 450 мс, 1B 1,9 с.
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
F='Loading|LOAD|Key  |-----|UNEXPECTED|Notes:|can be ignored|^$'
run() { python probe_scale.py "$@" 2>&1 | grep -vE "$F"; }
backup() { cp -r results_scale "backup/results_scale_$(date +%F)_$1"; echo "$(date +%T) бэкап $1"; }
pause() { echo "$(date +%T) перерыв $1 мин"; sleep $(( $1 * 60 )); }
# часть 1 (~1 ч 50 мин): 70M, 160M, 410M — обе среды
run --model EleutherAI/pythia-70m
run --model EleutherAI/pythia-160m
run --model EleutherAI/pythia-410m
backup part1; pause 20
# часть 2 (~1 ч 20 мин): 1B, основная среда, посевы 1-5
run --model EleutherAI/pythia-1b --envs flip-2_rev --first 1 --seeds 5
backup part2; pause 20
# часть 3 (~1 ч 20 мин): 1B, основная среда, посевы 6-10
run --model EleutherAI/pythia-1b --envs flip-2_rev --first 6 --seeds 10
backup part3; pause 15
# часть 4 (~30 мин): 1B, вторичная среда, 3 посева (описательно, пререгистрировано)
run --model EleutherAI/pythia-1b --envs flip-1_const --first 1 --seeds 3
backup final
echo "$(date +%T) ШКАЛА ГОТОВА"
