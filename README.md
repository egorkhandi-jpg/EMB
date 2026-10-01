# embodied-lm-stand — пакет для Claude Code

1. Распаковать в WSL2, например в `~/embodied-lm-stand`.
2. Windows: свежий драйвер NVIDIA; `C:\Users\<имя>\.wslconfig` с `[wsl2]` и `memory=24GB`, затем `wsl --shutdown`.
3. В Ubuntu: `bash setup_wsl.sh`.
4. В папке проекта запустить `claude`. Контекст и правила — в `CLAUDE.md`.

Команды Claude Code:
- `/check-env` — проверка окружения и короткий прогон
- `/precheck` — предварительные проверки среды
- `/series <модель> <посевы> <пауза>` — серия с контролями, мониторингом температуры и бэкапом
- `/summary [папка]` — сводка и механистический анализ
- `/next-step` — следующий шаг по плану (только план, без запуска)

Первый запуск: `/check-env`, затем `/series EleutherAI/pythia-70m 10 60`.
