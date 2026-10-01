"""Execute tutorial notebooks in fresh kernels, saving review artifacts elsewhere.

Run from any directory after installing .[examples]:
    python benchmark/check_notebooks.py --output /tmp/mageometry-notebooks

The tracked notebooks are never overwritten. Each notebook runs from its own
source directory, with its normal parameters and without skipping cells.
"""

import argparse
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='*', type=Path,
                        help='notebook paths (default: every tutorial)')
    parser.add_argument('--output', type=Path, required=True,
                        help='directory for executed notebooks')
    parser.add_argument('--timeout', type=int, default=180,
                        help='maximum seconds per cell (default: 180)')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    sources = sorted(path.resolve() for path in (args.notebooks or
                     (ROOT / 'examples/notebooks').rglob('*.ipynb')))
    if len({path.name for path in sources}) != len(sources):
        parser.error('input filenames must be distinct in the output directory')
    if any(output / path.name == path for path in sources):
        parser.error('output must not overwrite a source notebook')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib'))
    failures = []
    for path in sources:
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        print(f'Running {path.name}', flush=True)
        try:
            NotebookClient(notebook, timeout=args.timeout, kernel_name='python3',
                           resources={'metadata': {'path': str(path.parent)}}).execute()
        except Exception as exc:
            failures.append(path.name)
            print(f'FAIL {path.name}: {exc}', flush=True)
        else:
            print(f'PASS {path.name} ({time.monotonic() - started:.1f}s)', flush=True)
        finally:
            nbformat.write(notebook, output / path.name)
    print(f'{len(sources) - len(failures)}/{len(sources)} notebooks passed', flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
