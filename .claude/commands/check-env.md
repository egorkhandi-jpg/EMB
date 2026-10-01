Проверь окружение для стенда:
1. Активировано ли `~/stand-env` (если нет — активируй; если его нет — предложи запустить `setup_wsl.sh` и дождись подтверждения).
2. `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — видна ли RTX 5050 и сборка CUDA ≥ 12.8.
3. `nvidia-smi` — свободная видеопамять, температура, режим питания.
4. `free -g` — сколько памяти видит WSL (ожидается ~24 ГБ после `.wslconfig`).
5. Короткий прогон: `python stand_gpu.py --cond lm --gens 2 --pairs 4 --eps 1 --out results_smoke` и время на поколение.
Итог — коротко: что работает, что нет, что сделать.
