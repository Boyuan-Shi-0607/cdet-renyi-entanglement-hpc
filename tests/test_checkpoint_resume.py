#!/usr/bin/env python3
"""End-to-end resident resume check with exact final checkpoint comparison."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, default=root / 'Aphrodite')
args = parser.parse_args()
binary = args.binary.resolve()
base = (root / 'examples/3x3-resident/params.txt').read_text()
environment = {k: v for k, v in os.environ.items() if not k.startswith('CDET_')}
environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def run(folder, steps, mode, label):
    text = re.sub(r'^MCMCMC_PRODUCTION_STEPS=.*$', f'MCMCMC_PRODUCTION_STEPS={steps}', base, flags=re.M)
    text = re.sub(r'^MCMCMC_RESUME=.*$', f'MCMCMC_RESUME={mode}', text, flags=re.M)
    (folder / 'params.txt').write_text(text)
    with (folder / f'{label}.log').open('w') as log:
        result = subprocess.run([str(binary)], cwd=folder, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError((folder / f'{label}.log').read_text()[-5000:])


with tempfile.TemporaryDirectory(prefix='cdet-resume-') as temporary:
    directory = Path(temporary)
    uninterrupted, resumed = directory / 'uninterrupted', directory / 'resumed'
    uninterrupted.mkdir()
    resumed.mkdir()
    run(uninterrupted, 1500, 'off', 'full')
    run(resumed, 137, 'off', 'partial')
    run(resumed, 1500, 'require', 'resume')
    a = sorted((uninterrupted / 'mcmcmc_checkpoints').glob('chain_*.chk'))
    b = sorted((resumed / 'mcmcmc_checkpoints').glob('chain_*.chk'))
    assert len(a) == len(b) == 2
    for x, y in zip(a, b):
        assert x.read_bytes() == y.read_bytes(), f'Checkpoint differs: {x.name}'
        assert 'production_proposed_steps 1500\n' in x.read_text()
    print('PASS: 2 chains, 1500 uninterrupted steps == 137 + 1363 resumed steps (byte-identical checkpoints)')
