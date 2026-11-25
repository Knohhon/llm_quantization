#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python3 - <<'PY'
from common import *
quants = selected()
check_models(list(dict.fromkeys(['F16'] + quants)))
path, meta = start_run('size', quants)
baseline = model('F16').stat().st_size
rows = []
for q in quants:
    size = model(q).stat().st_size
    rows.append(dict(quant=q, size_bytes=size, size_mib=size / 2**20,
                     compression=baseline / size, reduction_pct=100 * (1 - size / baseline)))
save('size', rows, path, meta)
table([dict(quant=r['quant'], size=f"{r['size_mib']:.2f}", compression=f"{r['compression']:.2f}x",
            reduction=f"{r['reduction_pct']:.2f}%") for r in rows],
      [('quant', 'Quant'), ('size', 'Size (MiB)'), ('compression', 'Compression'), ('reduction', 'Reduction')])
PY
