Проверь окружение для стенда (нативный Windows, Git Bash; WSL не используется — см. CLAUDE.md):
1. `python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__)"` — видна ли RTX 5050 и сборка CUDA ≥ 12.8.
2. `nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total --format=csv` — свободная видеопамять, температура.
3. `export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1`, затем короткий прогон: `python stand_gpu.py --cond lm --gens 2 --pairs 4 --eps 1 --out results_smoke` и время на поколение. Папку `results_smoke` потом удалить.
4. `git status` — дерево чистое? Результаты, снятые с грязного дерева, помечаются в `result_*.json` (`git_dirty`).
Итог — коротко: что работает, что нет, что сделать.
