"""Run the unchanged diagnostic questions against an explicitly frozen stage snapshot."""
import argparse
import json
import re
from pathlib import Path

from evaluate_formal import DEFAULT, ROOT, digest, run, save


def prepare(directory):
    if directory.exists():
        raise SystemExit('Stage snapshot already exists; use a new stage name.')
    baseline = json.loads((DEFAULT / 'manifest.json').read_text(encoding='utf-8'))
    directory.mkdir(parents=True)
    (directory / 'gold.json').write_bytes((DEFAULT / 'gold.json').read_bytes())
    # The original manifest/results remain untouched. Each stage pins its own code.
    files = {name:digest(ROOT / name) for name in baseline['files']}
    for path in (ROOT / 'src').rglob('*.py'):
        files[path.relative_to(ROOT).as_posix()] = digest(path)
    for path in [Path(__file__), directory / 'gold.json', DEFAULT / 'manifest.json',
                 DEFAULT / 'run1/results.json', ROOT / 'data/eval/independent_v2/manifest.json']:
        files[path.relative_to(ROOT).as_posix()] = digest(path)
    save(directory / 'manifest.json', {
        'baseline_commit':baseline['baseline_commit'], 'files':files,
        'purpose':'36-case development diagnostics; not held-out accuracy',
        'baseline_manifest_sha256':digest(DEFAULT / 'manifest.json')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]+', args.stage):
        parser.error('stage must be a simple lowercase name')
    directory = ROOT / 'data/eval/stages' / args.stage
    if args.prepare:
        prepare(directory)
    else:
        run(directory, args.resume)


if __name__ == '__main__':
    main()
