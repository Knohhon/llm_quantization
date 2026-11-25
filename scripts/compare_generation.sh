#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if (( $# > 1 )); then
    echo 'Использование: bash scripts/compare_generation.sh "промпт"' >&2
    exit 1
fi
python3 - "$@" <<'PY'
from common import *
prompt = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_PROMPT
if not prompt.strip():
    fail('промпт не должен быть пустым.')
quants = selected()
exe, help_text = preflight('llama-cli', CLI_REQUIRED)
check_models(quants)
path, meta = start_run('generation', quants)
meta.update(prompt=prompt, chat_template='chatml', temperature=0, seed=number('SEED'))
(path / 'prompt.txt').write_text(prompt + '\n')
rows = []
for q in quants:
    out, err = execute(cli_command(exe, help_text, q, prompt, number('MAX_NEW_TOKENS')), path, q, meta)
    rate = speed(out + '\n' + err)
    # Полный stdout сохраняется без удаления частей потенциального ответа модели.
    (path / (q + '.txt')).write_text(out)
    rows.append(dict(quant=q, decode_ts=rate, response_file=str((path / (q + '.txt')).relative_to(ROOT))))
    print(f'\n=== {q} ===\n{out}\nDecode: {rate:.2f} tokens/s', flush=True)
save('generation', rows, path, meta)
print('Ответы и параметры: ' + str(path))
PY
