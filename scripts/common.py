"""Общие проверки и сохранение результатов; самостоятельной точки входа нет."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MANIFEST = json.loads((ROOT / 'scripts/models.json').read_text())
ORDER = list(MANIFEST['files'])
DEFAULT_PROMPT = 'Write a Python function for binary search. Include a short example.'
NUMERIC = ['THREADS', 'CONTEXT_SIZE', 'PROMPT_TOKENS', 'GENERATION_TOKENS',
           'MAX_NEW_TOKENS', 'REPETITIONS', 'PPL_CONTEXT_SIZE']
SETTINGS = NUMERIC + ['SEED', 'KV_CACHE_TYPE', 'LOAD_MODE', 'PPL_CHUNKS']


def fail(message):
    raise SystemExit('Ошибка: ' + message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected():
    values = os.environ.get('QUANTS', ' '.join(ORDER)).replace(',', ' ').split()
    if not values or len(values) != len(set(values)) or any(q not in ORDER for q in values):
        fail('QUANTS: ожидается неповторяющийся список из ' + ' '.join(ORDER))
    return values


def number(name):
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        fail(name + ' должен быть целым числом; запускайте через соответствующий .sh.')
    if name in NUMERIC and value <= 0:
        fail(name + ' должен быть больше нуля.')
    if name == 'PPL_CHUNKS' and value != -1 and value <= 0:
        fail('PPL_CHUNKS должен быть -1 или положительным числом.')
    return value


def model(q):
    return ROOT / 'models' / MANIFEST['files'][q]['name']


def check_models(quants):
    for q in quants:
        path, expected = model(q), MANIFEST['files'][q]
        if not path.is_file() or path.stat().st_size != expected['size']:
            fail(f'{path}: нет модели ожидаемого размера. Запустите bash scripts/download.sh.')
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                fail(f'{path}: неверная сигнатура GGUF.')
    # SHA-256 каждого файла проверяет download.sh. Не читаем гигабайты перед
    # измерением RSS, чтобы не менять состояние page cache самим экспериментом.


def clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    env.update(LC_ALL='C', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    return env


def binary(name):
    path = Path(os.environ['LLAMA_BIN_DIR']).expanduser() / name
    if not path.is_file() or not os.access(path, os.X_OK):
        fail(f'нет исполняемого файла {path}; проверьте LLAMA_BIN_DIR.')
    return str(path)


def preflight(name, flags):
    exe = binary(name)
    p = subprocess.run([exe, '--help'], capture_output=True, text=True, env=clean_env(), timeout=30)
    help_text = p.stdout + p.stderr
    if p.returncode:
        fail(name + ' --help завершился с ошибкой: ' + help_text[-2000:])
    for flag in flags:
        if not re.search(r'(?<![\w-])' + re.escape(flag) + r'(?![\w-])', help_text):
            fail(f'{name}: установленная версия не поддерживает {flag}. См. --help.')
    return exe, help_text


def cpu_flags(help_text, bench=False):
    flags = ['-ngl', '0', '-t', str(number('THREADS')), '--device', 'none',
             '-ctk', os.environ['KV_CACHE_TYPE'], '-ctv', os.environ['KV_CACHE_TYPE']]
    if '--load-mode' in help_text:
        if os.environ['LOAD_MODE'] not in ('mmap', 'none'):
            fail('LOAD_MODE должен быть mmap или none.')
        flags += ['--load-mode', os.environ['LOAD_MODE']]
        if '--lazy-mode' in help_text:
            flags += ['--lazy-mode', 'off']
    else:
        fail('Эти скрипты требуют --load-mode (проверено на build 12733).')
    flags += ['--no-kv-offload'] + (['1'] if bench else [])
    flags += ['--no-op-offload'] + (['1'] if bench else [])
    if '--offline' in help_text:
        flags += ['--offline']
    # У llama-bench отдельные тесты pp и tg: n_ctx = n_prompt + n_gen.
    if not bench:
        flags += ['-c', str(number('CONTEXT_SIZE')), '-tb', str(number('THREADS'))]
    return flags


CPU_REQUIRED = ['-ngl', '-t', '--device', '-ctk', '-ctv', '--load-mode',
                '--no-kv-offload', '--no-op-offload']
CLI_REQUIRED = CPU_REQUIRED + ['-c', '-tb', '-n', '-p', '--temp', '--seed', '--single-turn',
                               '--simple-io', '--no-display-prompt', '--chat-template', '--color', '--show-timings']


def identity():
    for key in NUMERIC + ['SEED', 'PPL_CHUNKS']:
        number(key)
    versions = {}
    for name in ['llama-cli', 'llama-bench', 'llama-perplexity']:
        path = Path(os.environ['LLAMA_BIN_DIR']) / name
        if path.is_file():
            p = subprocess.run([str(path), '--version'], capture_output=True, text=True,
                               env=clean_env(), timeout=30)
            versions[name] = {'version': (p.stdout + p.stderr).strip(), 'sha256': digest(path)}
    cpu = subprocess.run(['lscpu'], capture_output=True, text=True, env=clean_env(), check=True).stdout
    # Частоты в lscpu могут меняться: для сравнения используем только стабильные поля.
    stable_cpu = '\n'.join(line for line in cpu.splitlines() if re.match(
        r'^(Architecture|CPU\(s\)|On-line CPU|Vendor ID|Model name|Thread\(s\)|Core\(s\)|Socket\(s\)|Flags|Hypervisor vendor):', line))
    signature = {'settings': {k: os.environ[k] for k in SETTINGS}, 'cpu': stable_cpu,
                 'platform': platform.platform(), 'llama': versions,
                 'models': MANIFEST,
                 'corpus_sha256': digest(ROOT / 'data/perplexity.txt')}
    encoded = json.dumps(signature, sort_keys=True).encode()
    return signature, hashlib.sha256(encoded).hexdigest(), cpu


def start_run(kind, quants):
    signature, fingerprint, cpu = identity()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path = RESULTS / ('generations' if kind == 'generation' else 'raw/' + kind) / stamp
    path.mkdir(parents=True)
    metadata = {'experiment': kind, 'created_utc': stamp, 'quants': quants,
                'fingerprint': fingerprint, 'signature': signature, 'lscpu': cpu,
                'commands': [], 'raw_directory': str(path.relative_to(ROOT))}
    (path / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return path, metadata


def execute(command, path, q, metadata):
    print(f'[{q}] {shlex.join(map(str, command))}', file=sys.stderr, flush=True)
    metadata['commands'].append([str(x) for x in command])
    (path / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    out, err = path / (q + '.stdout.txt'), path / (q + '.stderr.txt')
    with out.open('w') as stdout, err.open('w') as stderr:
        p = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=clean_env())
    if p.returncode:
        fail(f'{q}: команда завершилась с кодом {p.returncode}; см. {err} и {out}.')
    return out.read_text(), err.read_text()


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        fail('некорректное числовое измерение: ' + str(value))
    return value


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def table(rows, columns):
    rendered = [[str(row.get(key, 'N/A')) for key, title in columns] for row in rows]
    widths = [max(len(title), *(len(row[i]) for row in rendered)) for i, (_, title) in enumerate(columns)]
    print('  '.join(title.ljust(widths[i]) for i, (_, title) in enumerate(columns)))
    print('  '.join('-' * width for width in widths))
    for row in rendered:
        print('  '.join(value.ljust(widths[i]) for i, value in enumerate(row)))


def save(kind, rows, path, metadata):
    # Архив сохраняет каждый запуск; текущий CSV заменяется целиком, без
    # подмешивания строк предыдущего запуска с другими настройками.
    write_csv(path / 'results.csv', rows)
    metadata['results_sha256'] = digest(path / 'results.csv')
    (path / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    write_csv(RESULTS / (kind + '.csv'), rows)
    (RESULTS / (kind + '.meta.json')).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


def speed(text):
    found = re.findall(r'Generation:\s*([\d.]+)\s*t/s', text)
    if not found:
        found = re.findall(r'(?m)^.*(?<!prompt )eval time\s*=.*?([\d.]+) tokens per second', text)
    if not found:
        fail('не удалось прочитать decode throughput; проверьте сохранённые логи.')
    return positive(found[-1])


def cli_command(exe, help_text, q, prompt, tokens):
    return [exe, '-m', str(model(q))] + cpu_flags(help_text) + [
        '-n', str(tokens), '-p', prompt, '--temp', '0', '--seed', str(number('SEED')),
        '--single-turn', '--simple-io', '--no-display-prompt', '--color', 'off',
        '--chat-template', 'chatml', '--show-timings']
