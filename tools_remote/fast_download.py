"""Параллельное скачивание модели HF по частям (на экземпляре; канал к HF ограничен ≈ 0,25 МБ/с на соединение — замер 2026-10-05).
Файлы — в ~/embodied-lm-stand/models/<имя>/ (config, токенизатор, safetensors); sha256 больших файлов сверяется с метаданными HF.

python tools_remote/fast_download.py EleutherAI/pythia-6.9b --threads 24
"""
import argparse, concurrent.futures as cf, hashlib, os, sys, time
import httpx
from huggingface_hub import HfApi

CHUNK = 64 * 1024 * 1024


def fetch_range(url, path, start, end, tries=8):
    for t in range(tries):
        try:
            pos = start
            with httpx.stream('GET', url, headers={'Range': f'bytes={start}-{end}'}, timeout=60, follow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    raise IOError(f'код {r.status_code}')
                with open(path, 'r+b') as f:
                    f.seek(pos)
                    for block in r.iter_bytes(1 << 20):
                        f.write(block); pos += len(block)
            if pos != end + 1:
                raise IOError(f'неполная часть {pos - start} из {end - start + 1}')
            return
        except Exception as e:                                                   # повтор части
            time.sleep(3 + 2 * t)
            last = e
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo')
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
    a = ap.parse_args()
    info = HfApi().model_info(a.repo, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith(('.json', '.safetensors', '.txt'))]
    d = os.path.join(a.out, a.repo.split('/')[-1]); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    for s in files:
        path = os.path.join(d, s.rfilename); size = s.size
        sha = s.lfs.sha256 if s.lfs else None
        if os.path.exists(path) and os.path.getsize(path) == size and (sha is None or os.path.exists(path + '.ok')):
            continue
        url = f'https://huggingface.co/{a.repo}/resolve/main/{s.rfilename}'
        with open(path, 'wb') as f:
            f.truncate(size)
        ranges = [(i, min(i + CHUNK, size) - 1) for i in range(0, size, CHUNK)] or [(0, 0)]
        if size == 0:
            continue
        with cf.ThreadPoolExecutor(a.threads) as ex:
            list(ex.map(lambda r: fetch_range(url, path, *r), ranges))
        if sha:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for block in iter(lambda: f.read(1 << 24), b''):
                    h.update(block)
            if h.hexdigest() != sha:
                print(f'НЕСОВПАДЕНИЕ sha256: {s.rfilename}', flush=True); sys.exit(1)
            open(path + '.ok', 'w').write(sha)
        print(f'{time.strftime("%H:%M:%S")} {a.repo}/{s.rfilename}: {size / 2**20:.0f} МБ, {time.time() - t0:.0f} с', flush=True)
    print(f'{a.repo}: готово, {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
