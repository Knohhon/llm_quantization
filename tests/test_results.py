"""Проверки расчётов на синтетических fixtures. Не являются измерениями LLM."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import common
import summary


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        defaults = dict(THREADS='4', CONTEXT_SIZE='2048', PROMPT_TOKENS='512',
                        GENERATION_TOKENS='128', MAX_NEW_TOKENS='256', REPETITIONS='3',
                        PPL_CONTEXT_SIZE='512', PPL_CHUNKS='-1', SEED='42',
                        KV_CACHE_TYPE='f16', LOAD_MODE='mmap', QUANTS='F16 Q4_K_M')
        self.env = patch.dict(os.environ, defaults)
        self.env.start()
        self.addCleanup(self.env.stop)

    def experiment(self, name, outputs=None):
        code = (common.ROOT / 'scripts' / name).read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        models = {}
        for q, size in [('F16', 1024), ('Q4_K_M', 256)]:
            models[q] = self.path / (q + '.gguf')
            models[q].write_bytes(b'G' * size)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(common, 'RESULTS', self.path))
            stack.enter_context(patch.object(common, 'check_models'))
            stack.enter_context(patch.object(common, 'model', side_effect=models.__getitem__))
            stack.enter_context(patch.object(common, 'preflight', return_value=('engine', '--load-mode')))
            stack.enter_context(patch.object(common, 'start_run', return_value=(self.path, {})))
            execute = stack.enter_context(patch.object(common, 'execute', side_effect=outputs))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            exec(compile(code, name, 'exec'), {})
        return execute

    def read(self, kind):
        with (self.path / (kind + '.csv')).open() as stream:
            return list(csv.DictReader(stream))

    def test_size_measures_actual_files_and_ratios(self):
        self.experiment('compare_size.sh')
        baseline, quant = self.read('size')
        self.assertEqual(float(baseline['compression']), 1)
        self.assertEqual(float(quant['compression']), 4)
        self.assertEqual(float(quant['reduction_pct']), 75)
        self.assertEqual(float(quant['size_mib']), 256 / 2**20)

    def test_benchmark_separates_prefill_and_decode(self):
        def raw(pp, tg):
            return json.dumps([dict(n_prompt=512, n_gen=0, avg_ts=pp, stddev_ts=2),
                               dict(n_prompt=0, n_gen=128, avg_ts=tg, stddev_ts=1)]), ''
        commands = self.experiment('benchmark.sh', [raw(100, 10), raw(150, 20)])
        baseline, quant = self.read('benchmark')
        self.assertEqual(float(quant['prefill_speedup']), 1.5)
        self.assertEqual(float(quant['decode_speedup']), 2)
        self.assertEqual(float(quant['decode_stddev']), 1)
        command = commands.call_args_list[0].args[0]
        self.assertEqual(command[command.index('-ngl') + 1], '0')
        self.assertEqual(command[command.index('--device') + 1], 'none')
        self.assertEqual(command[command.index('-r') + 1], '3')

    def test_perplexity_relative_change(self):
        self.experiment('perplexity.sh', [('', 'Final estimate: PPL = 10.0 +/- 0.1'),
                                         ('', 'Final estimate: PPL = 12.5 +/- 0.2')])
        self.assertEqual(float(self.read('perplexity')[1]['relative_change_pct']), 25)

    def test_memory_kib_to_mib_and_fixed_length(self):
        def run(command, path, quant, meta):
            (path / (quant + '.time.txt')).write_text(
                'Maximum resident set size (kbytes): 2048\n')
            return '', ''
        commands = self.experiment('compare_memory.sh', run)
        self.assertEqual(float(self.read('memory')[0]['peak_rss_mib']), 2)
        command = commands.call_args_list[0].args[0]
        self.assertIn('--ignore-eos', command)
        self.assertEqual(command[command.index('-n') + 1], '128')

    def test_generation_rate_formats(self):
        self.assertEqual(common.speed('[ Prompt: 10.0 t/s | Generation: 21.5 t/s ]'), 21.5)
        self.assertEqual(common.speed('llama_perf_context_print: eval time = 20 ms / 4 runs ( 200.0 tokens per second)'), 200)
        with self.assertRaises(SystemExit):
            common.speed('prompt eval time = 20 ms ( 200.0 tokens per second)')
        with self.assertRaises(SystemExit):
            common.positive('nan')

    def result_fixture(self, kind, fingerprint):
        source = self.path / (kind + '.csv')
        common.write_csv(source, [dict(quant='F16', decode_ts=10)])
        (self.path / (kind + '.meta.json')).write_text(json.dumps(dict(
            fingerprint=fingerprint, results_sha256=common.digest(source),
            signature=dict(models=common.MANIFEST))))

    def test_summary_missing_values(self):
        with patch.object(summary, 'RESULTS', self.path), contextlib.redirect_stdout(io.StringIO()):
            summary.main()
        self.assertEqual(len(self.read('summary')), 5)
        self.assertEqual(self.read('summary')[0]['perplexity'], 'N/A')
        self.assertTrue((self.path / 'summary.md').is_file())

    def test_summary_rejects_incompatible_settings(self):
        self.result_fixture('memory', 'threads4')
        self.result_fixture('benchmark', 'threads8')
        with patch.object(summary, 'RESULTS', self.path), self.assertRaisesRegex(SystemExit, 'несовместимые'):
            summary.main()

    def test_summary_rejects_tampered_csv(self):
        self.result_fixture('benchmark', 'threads4')
        with (self.path / 'benchmark.csv').open('a') as stream:
            stream.write('\n')
        with patch.object(summary, 'RESULTS', self.path), self.assertRaisesRegex(SystemExit, 'не соответствует'):
            summary.main()


if __name__ == '__main__':
    unittest.main()
