"""Раннер-очередь для длинных прогонов (правило 5 CLAUDE.md, ДОБАВЛЕНО 2026-10-02).

Зачем. Шкала размеров шла bash-скриптом: его нельзя править на ходу (bash читает скрипт по мере
выполнения), а части оказались в ~4 раза длиннее оценки — часть 1 шла ~8 ч без перерыва.
Здесь задачи берутся из файла очереди по одной, и перед каждой проверяются:
  - файл-флаг паузы (по умолчанию PAUSE в корне): пока он есть — ждать;
  - время с начала части: если ≥ --part-hours (2 ч) — перерыв --break-min (20 мин), новая часть;
  - температура GPU: выше 85 °C — ждать остывания ниже 70 °C.
Очередь можно дописывать и править во время работы: файл перечитывается перед каждой задачей.
Строка очереди — команда оболочки; '#' — комментарий. Выполненные строки пишутся в <очередь>.done
(повторно не запускаются), упавшие — в <очередь>.failed. Задачи должны быть короче части
(дробить по посевам: скрипты стенда досчитывают недостающее).

python run_queue.py queue.txt [--part-hours 2] [--break-min 20] [--pause-file PAUSE]
Остановить между задачами: создать файл STOP (раннер выйдет перед следующей задачей).
"""
import argparse, os, subprocess, sys, time


def temp():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('queue')
    ap.add_argument('--part-hours', type=float, default=2.0)
    ap.add_argument('--break-min', type=float, default=20.0)
    ap.add_argument('--pause-file', default='PAUSE')
    ap.add_argument('--stop-file', default='STOP')
    a = ap.parse_args()
    done_path, fail_path, log_path = a.queue + '.done', a.queue + '.failed', a.queue + '.log'
    env = dict(os.environ, PYTHONIOENCODING='utf-8', HF_HUB_DISABLE_SYMLINKS_WARNING='1')

    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            print(line, file=f)

    part_start = time.time()
    log(f'старт: часть ≤ {a.part_hours} ч, перерыв {a.break_min} мин')
    while True:
        if os.path.exists(a.stop_file):
            log(f'найден {a.stop_file} — выход между задачами'); break
        if os.path.exists(a.pause_file):
            log(f'пауза ({a.pause_file})')
            while os.path.exists(a.pause_file) and not os.path.exists(a.stop_file):
                time.sleep(30)
            log('пауза снята'); part_start = time.time(); continue
        lines = [l.strip() for l in open(a.queue, encoding='utf-8') if l.strip() and not l.strip().startswith('#')]
        done = set(l.strip() for l in open(done_path, encoding='utf-8')) if os.path.exists(done_path) else set()
        failed = set(l.strip() for l in open(fail_path, encoding='utf-8')) if os.path.exists(fail_path) else set()
        todo = [l for l in lines if l not in done and l not in failed]
        if not todo:
            log('очередь пуста — готово'); break
        if time.time() - part_start >= a.part_hours * 3600:
            log(f'часть длилась {(time.time() - part_start) / 3600:.2f} ч — перерыв {a.break_min} мин')
            time.sleep(a.break_min * 60); part_start = time.time(); continue
        t = temp()
        if t is not None and t > 85:
            log(f'GPU {t} °C — остывание')
            while (t := temp()) is not None and t >= 70:
                time.sleep(30)
        cmd = todo[0]
        log(f'GPU {t} °C | задача: {cmd}')
        t0 = time.time()
        rc = subprocess.run(cmd, shell=True, env=env).returncode
        with open(done_path if rc == 0 else fail_path, 'a', encoding='utf-8') as f:
            print(cmd, file=f)
        log(f'{"готово" if rc == 0 else f"ОШИБКА rc={rc}"} за {time.time() - t0:.0f} с: {cmd}')


if __name__ == '__main__':
    main()
