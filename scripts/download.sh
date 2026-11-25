#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python3 - <<'PY'
import json
import urllib.request
import urllib.error
from common import MANIFEST, ROOT, selected, model, digest, fail

quants = selected()
pending = []
for q in quants:
    p, f = model(q), MANIFEST['files'][q]
    if p.is_file() and p.stat().st_size == f['size'] and digest(p) == f['sha256']:
        print(f'[{q}] SHA-256 верен, пропуск.', flush=True)
    else:
        pending.append(q)
if not pending:
    raise SystemExit(0)
repo, rev = MANIFEST['repository'], MANIFEST['revision']
try:
    # Проверяем опубликованные точные имена и LFS-метаданные до загрузки весов.
    url = f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true'
    with urllib.request.urlopen(url, timeout=30) as response:
        remote = json.load(response)
    if remote['sha'] != rev:
        fail('revision API не совпала с зафиксированной.')
    files = {x['rfilename']: x for x in remote['siblings']}
    for q in pending:
        expected = MANIFEST['files'][q]
        actual = files.get(expected['name'], {})
        if (actual.get('size') != expected['size'] or
                actual.get('lfs', {}).get('sha256') != expected['sha256']):
            fail(f'{q}: имя, размер или SHA-256 опубликованного файла не совпали с manifest.')
    for q in pending:
        p, f = model(q), MANIFEST['files'][q]
        p.parent.mkdir(parents=True, exist_ok=True)
        part = p.with_suffix('.gguf.part')
        print(f'[{q}] загрузка {f["name"]}, {f["size"] / 2**20:.1f} MiB...', flush=True)
        url = f'https://huggingface.co/{repo}/resolve/{rev}/{f["name"]}'
        with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as stream:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
        if part.stat().st_size != f['size'] or digest(part) != f['sha256']:
            fail(f'{q}: загруженный файл не прошёл проверку размера/SHA-256; оставлен {part}.')
        part.replace(p)
        print(f'[{q}] сохранён, SHA-256 верен.', flush=True)
except (OSError, urllib.error.URLError, ValueError, KeyError) as error:
    fail(f'загрузка Hugging Face: {error}. Проверьте сеть и повторите download.sh. '
         'Незавершённые .part не считаются моделями.')
PY
