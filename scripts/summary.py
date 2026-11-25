#!/usr/bin/env python3
"""Сводка только сохранённых измерений; не запускает эксперименты."""
import csv
import json
from common import ORDER, RESULTS, MANIFEST, digest, fail, table, write_csv


def main():
    data = {}
    reference = None
    for kind in ['size', 'memory', 'benchmark', 'perplexity']:
        source = RESULTS / (kind + '.csv')
        if not source.exists():
            data[kind] = {}
            continue
        meta_path = RESULTS / (kind + '.meta.json')
        if not meta_path.is_file():
            fail(f'{source}: нет метаданных; повторите соответствующий эксперимент.')
        meta = json.loads(meta_path.read_text())
        if meta.get('results_sha256') != digest(source):
            fail(f'{source}: CSV не соответствует метаданным; повторите эксперимент.')
        if meta['signature']['models'] != MANIFEST:
            fail(f'{source}: другая revision моделей.')
        if kind != 'size':
            if reference and reference != meta['fingerprint']:
                fail('несовместимые настройки результатов. Повторите memory, benchmark и perplexity '
                     'с одним config.env и одинаковыми переменными окружения; архивы находятся в results/raw/.')
            reference = meta['fingerprint']
        with source.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if any(r['quant'] not in ORDER for r in rows) or len({r['quant'] for r in rows}) != len(rows):
            fail(f'{source}: неизвестная или повторяющаяся квантизация.')
        data[kind] = {r['quant']: r for r in rows}

    def metric(kind, q, field):
        return data[kind].get(q, {}).get(field, 'N/A')

    rows = [dict(quant=q, size_mib=metric('size', q, 'size_mib'),
                 peak_rss_mib=metric('memory', q, 'peak_rss_mib'),
                 prefill_ts=metric('benchmark', q, 'prefill_ts'),
                 decode_ts=metric('benchmark', q, 'decode_ts'),
                 decode_speedup=metric('benchmark', q, 'decode_speedup'),
                 perplexity=metric('perplexity', q, 'perplexity')) for q in ORDER]
    columns = [('quant', 'Quant'), ('size_mib', 'Size MiB'), ('peak_rss_mib', 'Peak RSS MiB'),
               ('prefill_ts', 'Prefill t/s'), ('decode_ts', 'Decode t/s'),
               ('decode_speedup', 'Speedup'), ('perplexity', 'PPL')]
    write_csv(RESULTS / 'summary.csv', rows)
    display = []
    for row in rows:
        display.append({key: (value if key == 'quant' or value == 'N/A' else
                              f'{float(value):.2f}' + ('x' if key == 'decode_speedup' else ''))
                        for key, value in row.items()})
    table(display, columns)
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(row[key] for key, _ in columns) + ' |' for row in display]
    (RESULTS / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
