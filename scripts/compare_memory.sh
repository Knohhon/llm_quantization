#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python3 - <<'PY'
from common import *
quants = selected()
exe, help_text = preflight('llama-cli', CLI_REQUIRED + ['--ignore-eos'])
if not os.access('/usr/bin/time', os.X_OK):
    fail('нужен GNU time по пути /usr/bin/time.')
check_models(quants)
path, meta = start_run('memory', quants)
meta['prompt'] = DEFAULT_PROMPT
rows = []
for q in quants:
    command = ['/usr/bin/time', '-v', '-o', str(path / (q + '.time.txt'))] + cli_command(
        exe, help_text, q, DEFAULT_PROMPT, number('GENERATION_TOKENS')) + ['--ignore-eos']
    execute(command, path, q, meta)
    report = (path / (q + '.time.txt')).read_text()
    match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', report)
    if not match:
        fail('не удалось прочитать Peak RSS из ' + str(path))
    rows.append(dict(quant=q, file_size_mib=model(q).stat().st_size / 2**20,
                     peak_rss_mib=positive(match[1]) / 1024))
save('memory', rows, path, meta)
table([dict(quant=r['quant'], size=f"{r['file_size_mib']:.2f}", rss=f"{r['peak_rss_mib']:.2f}") for r in rows],
      [('quant', 'Quant'), ('size', 'File size (MiB)'), ('rss', 'Peak RSS (MiB)')])
PY
