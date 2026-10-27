"""Freeze a fresh development candidate without opening the held-out v2 set."""
import argparse
import difflib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

from evaluate_formal import ROOT, digest, run, save


def prepare(directory, baseline):
    if directory.exists():
        raise SystemExit('Candidate already exists; use a new stage name.')
    frozen=json.loads((baseline/'manifest.json').read_text(encoding='utf-8'))
    directory.mkdir(parents=True)
    (directory/'gold.json').write_bytes((baseline/'gold.json').read_bytes())
    names={name for name in frozen['files'] if 'independent_v2' not in name
        and not name.replace('\\','/').startswith('data/eval/stages/')}
    for folder in ['src','tests']:
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*.py'))
    names.update([Path(__file__).relative_to(ROOT).as_posix(),
        (directory/'gold.json').relative_to(ROOT).as_posix()])
    git=['git','-c','safe.directory='+ROOT.as_posix()]
    paths=['src','tests','pyproject.toml',Path(__file__).relative_to(ROOT).as_posix()]
    patch=subprocess.check_output([*git,'diff','--',*paths],cwd=ROOT)
    added=subprocess.check_output([*git,'ls-files','--others','--exclude-standard','--',*paths],cwd=ROOT,text=True)
    for name in added.splitlines():
        patch+=''.join(difflib.unified_diff([], (ROOT/name).read_text(encoding='utf-8').splitlines(True),
            fromfile='/dev/null',tofile='b/'+name)).encode('utf-8')
    (directory/'candidate.patch').write_bytes(patch)
    packages={}
    for package in ['torch','transformers','sentence-transformers','numpy','pytest','lm-format-enforcer']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    save(directory/'manifest.json',{'baseline_commit':subprocess.check_output(
        [*git,'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'purpose':'Development diagnostics, not held-out accuracy; independent_v2 unopened',
        'baseline_results_sha256':digest(baseline/'run1/results.json'), 'packages':packages,
        'files':{name:digest(ROOT/name) for name in sorted(names)}})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'data/eval/stages/reliability_v1')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch('[a-z0-9_-]+',args.stage):parser.error('Invalid stage name')
    directory=ROOT/'data/eval/stages'/args.stage
    if args.prepare:prepare(directory,args.baseline)
    else:run(directory,args.resume)


if __name__=='__main__':main()
