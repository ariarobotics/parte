import argparse
import ctypes
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[2]
WEBSITE = SOURCE / "website"
DATASET_NAMES = {"3DMatch": "3DMatch", "3DLoMatch": "3DLoMatch", "kitti-10m": "KITTI-10m", "kitti-lc": "KITTI-LC", "ETH": "ETH", "RESSO": "RESSO P2F"}


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def transforms(path):
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return [(int(lines[i][0]), int(lines[i][1]),
             [float(value) for row in lines[i + 1:i + 5] for value in row])
            for i in range(0, len(lines), 5)]


def collect(root, manifests):
    tasks = []
    option_parser = argparse.ArgumentParser(add_help=False)
    option_parser.add_argument("--segment-ground", action="store_true")
    option_parser.add_argument("--success-criteria", type=float, nargs=2)
    for name in manifests:
        for line in (root / name).read_text().splitlines():
            fields = shlex.split(line, comments=True)
            if not fields:
                continue
            scans, gt = root / fields[0], root / fields[1]
            rel = gt.relative_to(root / "benchmarks")
            dataset, sequence = rel.parts[0], Path(*rel.parts[1:-1]).as_posix()
            options = option_parser.parse_args(fields[2:])
            voxel = float((scans / "voxel_size").read_text())
            ranges = {}
            for path in sorted(gt.parent.glob("gt_*_*.log")):
                _, start, end = path.stem.split("_")
                for src, tgt, _ in transforms(path):
                    ranges[src, tgt] = f"{start}-{end} m"
            
            for src, tgt, matrix in transforms(gt):
                tasks.append(dict(
                    dataset=DATASET_NAMES[dataset], sequence=sequence,
                    sourceScan=src, targetScan=tgt,
                    source=scans / f"cloud_bin_{src}.ply",
                    target=scans / f"cloud_bin_{tgt}.ply", voxel=voxel,
                    segmentGround=options.segment_ground, criteria=options.success_criteria,
                    groundTruth=matrix, distanceRange=ranges.get((src, tgt))
                ))
    return tasks


def errors(estimate, target_to_source):
    r = [[target_to_source[4*j+i] for j in range(3)] for i in range(3)]
    t = [-sum(r[i][j] * target_to_source[4*j+3] for j in range(3)) for i in range(3)]
    translation = math.sqrt(sum((estimate[4*i+3]-t[i])**2 for i in range(3)))
    trace = sum(estimate[4*i+j] * r[i][j] for i in range(3) for j in range(3))
    rotation = math.degrees(math.acos(max(-1.0, min(1.0, (trace-1)/2))))
    return translation, rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets-root', type=Path, default=Path('/mnt/processed_datasets'))
    parser.add_argument('--output', type=Path, default=WEBSITE / 'cache/viewer')
    parser.add_argument('--manifest', action='append')
    args = parser.parse_args()
    root, output = args.datasets_root.resolve(), args.output.resolve()
    tasks = collect(root, args.manifest or ['indoor.txt', 'outdoor.txt', 'additional.txt'])
    library = {'linux': 'libparte-cache.so', 'darwin': 'libparte-cache.dylib', 'win32': 'parte-cache.dll'}[sys.platform]
    export_pair = ctypes.CDLL(str(SOURCE / 'build' / library)).export_pair
    export_pair.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_bool, ctypes.c_char_p]
    export_pair.restype = None

    (output / 'scans').mkdir(parents=True, exist_ok=True)
    (output / 'pairs').mkdir(parents=True, exist_ok=True)
    manifest, summary = {}, {}
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for index, task in enumerate(tasks):
            pair_id = f'{index:x}'
            export_pair(str(task['source']).encode(), str(task['target']).encode(),
                        task['voxel'], task['segmentGround'], str(work).encode())
            trace = json.loads((work / 'trace.json').read_text())
            scans = []
            for role in ('source', 'target'):
                scan_id = re.sub(r'[^a-zA-Z0-9_-]+', '-', f"{task['dataset']}-{task['sequence']}-{task[role + 'Scan']}")
                shutil.copyfile(work / f'{role}.bin', output / 'scans' / f'{scan_id}.bin')
                scans.append(scan_id)
            translation, rotation = errors(trace['transform'], task['groundTruth'])
            metadata = dict(source=scans[0], target=scans[1], transform=trace['transform'][:12],
                            runtime=[trace['registrationRuntimeMs'], trace['loadingRuntimeMs']],
                            errors=[translation, rotation])
            destination = output / 'pairs' / pair_id
            destination.mkdir(exist_ok=True)
            shutil.copyfile(work / 'matches.bin', destination / 'matches.bin')
            write_json(destination / 'metadata.json', metadata)
            row = [task['sourceScan'], task['targetScan'], pair_id]
            if task['distanceRange']:
                row.append(task['distanceRange'])
            manifest.setdefault(task['dataset'], {}).setdefault(task['sequence'], []).append(row)
            totals = summary.setdefault(task['dataset'], dict(pairs=0, successful=0))
            totals['pairs'] += 1
            totals['successful'] += translation <= task['criteria'][0] and rotation <= task['criteria'][1]
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(tasks):
                print(f'[{index + 1}/{len(tasks)}] {time.monotonic() - started:.1f}s', flush=True)

    write_json(output / 'manifest.json', manifest)
    print(f'Compact cache ready: {output}\n' + json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
