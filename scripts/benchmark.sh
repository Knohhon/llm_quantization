#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python3 - <<'PY'
from common import *
quants = selected()
exe, help_text = preflight('llama-bench', CPU_REQUIRED + ['-p', '-n', '-r', '-o'])
check_models(quants)
path, meta = start_run('benchmark', quants)
rows = []
for q in quants:
    command = [exe, '-m', str(model(q))] + cpu_flags(help_text, bench=True) + [
        '-p', str(number('PROMPT_TOKENS')), '-n', str(number('GENERATION_TOKENS')),
        '-r', str(number('REPETITIONS')), '-o', 'json']
    out, err = execute(command, path, q, meta)
    (path / (q + '.json')).write_text(out)
    try:
        values = json.loads(out)
        prefill = [v for v in values if v['n_prompt'] == number('PROMPT_TOKENS') and v['n_gen'] == 0]
        decode = [v for v in values if v['n_gen'] == number('GENERATION_TOKENS') and v['n_prompt'] == 0]
        if len(prefill) != 1 or len(decode) != 1:
            fail(f'{q}: ожидалось ровно два отдельных теста pp/tg; см. {path}.')
        pp, tg = prefill[0], decode[0]
        pp_sd, tg_sd = float(pp['stddev_ts']), float(tg['stddev_ts'])
        if any(not math.isfinite(x) or x < 0 for x in [pp_sd, tg_sd]):
            fail(f'{q}: неверное стандартное отклонение.')
        rows.append(dict(quant=q, prefill_ts=positive(pp['avg_ts']), prefill_stddev=pp_sd,
                         decode_ts=positive(tg['avg_ts']), decode_stddev=tg_sd))
    except (ValueError, TypeError, KeyError) as error:
        fail(f'неизвестный JSON llama-bench: {error}; см. {path}.')
baseline = next((r for r in rows if r['quant'] == 'F16'), None)
for row in rows:
    row['prefill_speedup'] = row['prefill_ts'] / baseline['prefill_ts'] if baseline else 'N/A'
    row['decode_speedup'] = row['decode_ts'] / baseline['decode_ts'] if baseline else 'N/A'
save('benchmark', rows, path, meta)
table([dict(quant=r['quant'], pp=f"{r['prefill_ts']:.2f} ± {r['prefill_stddev']:.2f}",
            tg=f"{r['decode_ts']:.2f} ± {r['decode_stddev']:.2f}",
            speedup=f"{r['decode_speedup']:.2f}x" if baseline else 'N/A') for r in rows],
      [('quant', 'Quant'), ('pp', 'Prefill t/s ± SD'), ('tg', 'Decode t/s ± SD'), ('speedup', 'Decode speedup')])
PY
