#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python3 - <<'PY'
from common import *
quants = selected()
exe, help_text = preflight('llama-perplexity', CPU_REQUIRED + ['-c', '-tb', '-f', '--chunks'])
check_models(quants)
path, meta = start_run('perplexity', quants)
rows = []
for q in quants:
    flags = cpu_flags(help_text)
    flags[flags.index('-c') + 1] = str(number('PPL_CONTEXT_SIZE'))
    command = [exe, '-m', str(model(q)), '-f', str(ROOT / 'data/perplexity.txt')] + flags + [
        '--chunks', str(number('PPL_CHUNKS'))]
    out, err = execute(command, path, q, meta)
    matches = re.findall(r'Final estimate:\s*PPL\s*=\s*([\d.eE+\-]+)', out + '\n' + err)
    if len(matches) != 1:
        fail(f'{q}: нет однозначной итоговой PPL. См. {path}; корпус должен содержать >= 2 контекстов.')
    rows.append(dict(quant=q, perplexity=positive(matches[0])))
baseline = next((r['perplexity'] for r in rows if r['quant'] == 'F16'), None)
for row in rows:
    row['relative_change_pct'] = 100 * (row['perplexity'] / baseline - 1) if baseline else 'N/A'
save('perplexity', rows, path, meta)
table([dict(quant=r['quant'], ppl=f"{r['perplexity']:.4f}",
            delta=f"{r['relative_change_pct']:+.2f}%" if baseline else 'N/A') for r in rows],
      [('quant', 'Quant'), ('ppl', 'Perplexity'), ('delta', 'Change vs F16')])
PY
