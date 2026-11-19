"""Write datasets.json for the four tasks from a directory holding the HDF5 files.

    python prepare_datasets.py --root /path/to/datasets --output datasets.json
Expected files: pusht_expert_train.h5, cube_single_expert.h5, reacher.h5, tworoom.h5 (searched recursively)."""
import argparse
import json
from pathlib import Path

FILES = dict(pusht='pusht_expert_train.h5', cube='cube_single_expert.h5', reacher='reacher.h5',
             tworoom='tworoom.h5')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('datasets.json'))
    a = p.parse_args()
    paths = {}
    for task, name in FILES.items():
        hits = sorted(a.root.resolve().rglob(name))
        if not hits:
            raise SystemExit(f'{name} not found under {a.root}')
        paths[task] = dict(train=str(hits[0]), evaluation=str(hits[0]))
    a.output.write_text(json.dumps(paths, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
