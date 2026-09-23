
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[2]
WEBSITE = SOURCE / "website"
HERE = WEBSITE / "cache"
DATASET_NAMES = {"3DMatch": "3DMatch", "3DLoMatch": "3DLoMatch", "kitti-10m": "KITTI-10m", "kitti-lc": "KITTI-LC", "ETH": "ETH", "RESSO": "RESSO P2F"}


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def checked_path(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError(f"missing file or path outside dataset root: {relative}")
    return result


def transforms(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or len(lines) % 5:
        raise ValueError(f"expected five lines per transform: {path}")
    result = []
    for i in range(0, len(lines), 5):
        head = lines[i].split()
        if len(head) < 2:
            raise ValueError(f"invalid pair header: {path}")
        matrix = [float(v) for line in lines[i + 1:i + 5] for v in line.split()]
        if len(matrix) != 16 or not all(math.isfinite(v) for v in matrix):
            raise ValueError(f"invalid transform: {path}")
        if any(abs(matrix[12+j]-[0, 0, 0, 1][j]) > 1e-6 for j in range(4)):
            raise ValueError(f"invalid homogeneous transform: {path}")
        result.append((int(head[0]), int(head[1]), matrix))
    return result


def collect(root, manifests):
    tasks, protocol_hashes, seen = [], {}, set()
    option_parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    option_parser.add_argument("--segment-ground", action="store_true")
    option_parser.add_argument("--success-criteria", type=float, nargs=2, required=True)
    for name in manifests:
        manifest = checked_path(root, name)
        protocol_hashes[str(manifest.relative_to(root))] = digest(manifest)
        for line in manifest.read_text().splitlines():
            fields = shlex.split(line, comments=True)
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"invalid manifest line: {line}")
            scans = (root / fields[0]).resolve()
            if not scans.is_relative_to(root) or not scans.is_dir():
                raise ValueError(f"invalid scan directory: {fields[0]}")
            gt = checked_path(root, fields[1])
            rel = gt.relative_to(root / "benchmarks")
            dataset, sequence = rel.parts[0], Path(*rel.parts[1:-1]).as_posix()
            if dataset not in DATASET_NAMES:
                raise ValueError(f"unknown benchmark: {dataset}")
            options = option_parser.parse_args(fields[2:])
            criteria = options.success_criteria
            if not all(math.isfinite(v) and v > 0 for v in criteria):
                raise ValueError("success criteria must be finite and positive")
            voxel_path = checked_path(root, str((scans / "voxel_size").relative_to(root)))
            voxel = float(voxel_path.read_text())
            if not math.isfinite(voxel) or voxel <= 0:
                raise ValueError("voxel size must be finite and positive")
            protocol_hashes[str(gt.relative_to(root))] = digest(gt)
            protocol_hashes[str(voxel_path.relative_to(root))] = digest(voxel_path)
            ranges = {}
            for path in sorted(gt.parent.glob("gt_*_*.log")):
                match = re.fullmatch(r"gt_(\d+)_(\d+)\.log", path.name)
                if not match:
                    continue
                protocol_hashes[str(path.relative_to(root))] = digest(path)
                label = f"{match[1]}-{match[2]} m"
                for src, tgt, _ in transforms(path):
                    if (src, tgt) in ranges and ranges[src, tgt] != label:
                        raise ValueError(f"overlapping distance partitions: {path}")
                    ranges[src, tgt] = label
            for src, tgt, matrix in transforms(gt):
                slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", f"{dataset}-{sequence}-{src}-{tgt}").lower()
                if slug in seen:
                    raise ValueError(f"duplicate pair: {slug}")
                seen.add(slug)
                source = checked_path(root, str((scans / f"cloud_bin_{src}.ply").relative_to(root)))
                target = checked_path(root, str((scans / f"cloud_bin_{tgt}.ply").relative_to(root)))
                tasks.append(dict(slug=slug, dataset=DATASET_NAMES[dataset], sequence=sequence,
                                  sourceScan=src, targetScan=tgt, source=str(source.relative_to(root)),
                                  target=str(target.relative_to(root)), voxel=voxel,
                                  segmentGround=options.segment_ground, criteria=criteria,
                                  groundTruth=matrix, distanceRange=ranges.get((src, tgt))))
    if not tasks:
        raise ValueError("no benchmark pairs selected")
    return tasks, protocol_hashes


def errors(estimate, target_to_source):
    r = [[target_to_source[4*j+i] for j in range(3)] for i in range(3)]
    t = [-sum(r[i][j] * target_to_source[4*j+3] for j in range(3)) for i in range(3)]
    translation = math.sqrt(sum((estimate[4*i+3]-t[i])**2 for i in range(3)))
    trace = sum(estimate[4*i+j] * r[i][j] for i in range(3) for j in range(3))
    rotation = math.degrees(math.acos(max(-1.0, min(1.0, (trace-1)/2))))
    return translation, rotation


DATASET_SETTINGS = {
    "3DMatch": (0.05, False, 0.3, 15.0),
    "3DLoMatch": (0.05, False, 0.3, 15.0),
    "KITTI-10m": (0.3, True, 0.6, 5.0),
    "KITTI-LC": (0.3, True, 2.0, 5.0),
    "ETH": (0.1, False, 0.3, 15.0),
    "RESSO P2F": (0.05, False, 0.3, 15.0),
}


def publish_scan(data, output):
    sha = hashlib.sha256(data).hexdigest()
    scan_id = sha[:12]
    path = output / 'scans' / (scan_id + '.bin')
    if path.exists():
        if digest(path) != sha: raise ValueError('short scan ID collision or corrupted scan')
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            f.write(data)
        Path(f.name).replace(path)
    return scan_id, sha


def save_pair(task, trace, scans, matches, output, state, pair_id, key):
    if len(trace['transform']) != 16 or not all(math.isfinite(v) for v in trace['transform']):
        raise ValueError('invalid estimated transform')
    translation, rotation = errors(trace['transform'],task['groundTruth'])
    metadata=dict(source=scans[0][0],target=scans[1][0],transform=trace['transform'][:12],
                  runtime=[trace['registrationRuntimeMs'],trace['loadingRuntimeMs']],
                  errors=[translation,rotation])
    if not trace['completed']: metadata['error']=trace.get('error') or 'registration failed'
    destination=output/'pairs'/pair_id
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'matches.bin').write_bytes(matches)
    write_json(destination/'metadata.json',metadata)
    checksums={f'scans/{scan_id}.bin':sha for scan_id,sha in scans}
    checksums[f'pairs/{pair_id}/matches.bin']=digest(destination/'matches.bin')
    checksums[f'pairs/{pair_id}/metadata.json']=digest(destination/'metadata.json')
    write_json(state/'pairs'/f'{pair_id}.json',dict(key=key,checksums=checksums))
    return metadata


def process(task, root, output, state, executable, threads, key, pair_id):
    with tempfile.TemporaryDirectory(prefix='pair-',dir=state) as work:
        work=Path(work)
        subprocess.run([str(executable),str(root/task['source']),str(root/task['target']),
                        str(task['voxel']),str(int(task['segmentGround'])),str(work),str(threads)],
                       check=True,capture_output=True,text=True)
        trace=json.loads((work/'trace.json').read_text())
        scans=[publish_scan((work/f'{role}.bin').read_bytes(),output) for role in ('source','target')]
        return save_pair(task,trace,scans,(work/'matches.bin').read_bytes(),output,state,pair_id,key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root',type=Path,default=Path('/mnt/processed_datasets'))
    parser.add_argument('--output',type=Path,default=WEBSITE/'cache/viewer')
    parser.add_argument('--executable',type=Path,default=SOURCE/'build/parte-cache',
                        help='path to the separately compiled exporter')
    parser.add_argument('--manifest',action='append')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--rebuild',action='store_true',help='replace only this formatter-owned output')
    args=parser.parse_args()
    if min(args.workers,args.threads)<1: parser.error('counts must be positive')
    root,output,executable=args.datasets_root.resolve(),args.output.resolve(),args.executable.resolve()
    if not executable.is_file():
        parser.error(f"exporter not found: {executable}; build the parte-cache CMake target first")
    state=output.with_name(output.name+'.state')
    for destination in (output,state):
        if destination in (Path('/'),Path.home(),WEBSITE) or destination in WEBSITE.parents:
            parser.error('unsafe output directory')
        for protected in (root,WEBSITE/'src',WEBSITE/'public',HERE/'export.cpp',HERE/'build_cache.py',HERE/'CMakeLists.txt',executable.parent,
                          SOURCE/'common',SOURCE/'registration',SOURCE/'segmentation',SOURCE/'clipperp',SOURCE/'.git'):
            if protected and (destination.is_relative_to(protected) or protected.is_relative_to(destination)):
                parser.error('output overlaps source, input, or build')
    tasks,protocols=collect(root,args.manifest or ['indoor.txt','outdoor.txt','additional.txt'])
    for task in tasks:
        if (task['voxel'],task['segmentGround'],*task['criteria'])!=DATASET_SETTINGS[task['dataset']]:
            raise ValueError('dataset settings changed; update DATASET_SETTINGS')
    source_paths = [SOURCE/'CMakeLists.txt', SOURCE/'benchmark.cpp',
                    HERE/'CMakeLists.txt', HERE/'export.cpp', HERE/'build_cache.py']
    for name in ('common', 'registration', 'segmentation', 'cmake', 'clipperp'):
        source_paths.extend(p for p in (SOURCE/name).rglob('*')
                            if p.is_file() and p.suffix in ('.h','.hpp','.tpp','.cpp','.py','.txt','.cmake')
                            and '.git' not in p.parts)
    source_files = {str(p.relative_to(SOURCE)): digest(p) for p in sorted(source_paths)}
    executable_hash = digest(executable)
    config=dict(sourceHash=signature(source_files),protocolHash=signature(protocols),taskHash=signature(tasks),
                sourceCommit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
                threads=args.threads,executableHash=executable_hash,output=str(output))
    marker=state/'build.json'
    if (output.exists() and any(output.iterdir())) or (state.exists() and any(state.iterdir())):
        if not marker.is_file(): parser.error('output/state is not owned by this formatter')
        previous=json.loads(marker.read_text())
        if previous.get('output')!=str(output): parser.error('state belongs to a different output')
        if args.rebuild:
            if output.exists(): shutil.rmtree(output)
            shutil.rmtree(state)
        elif previous!=config: parser.error('source/configuration changed; use a new --output or --rebuild')
    (output/'scans').mkdir(parents=True,exist_ok=True);(state/'pairs').mkdir(parents=True,exist_ok=True)
    write_json(marker,config)
    inputs=sorted({task[role] for task in tasks for role in ('source','target')})
    input_hashes={path:digest(root/path) for path in inputs}
    results={};pending=[];started=time.monotonic();reused=0
    def accept(index,metadata,cached=False):
        nonlocal reused
        results[index]=metadata;reused+=cached
        if len(results)==1 or len(results)%100==0 or len(results)==len(tasks):
            print(f'[{len(results)}/{len(tasks)}] {time.monotonic()-started:.1f}s, reused {reused}',flush=True)
    for index,task in enumerate(tasks):
        pair_id=f'{index:x}'
        key=signature([config,task,input_hashes[task['source']],input_hashes[task['target']],executable_hash])
        resume=state/'pairs'/f'{pair_id}.json'
        if resume.is_file():
            saved=json.loads(resume.read_text())
            if saved['key']==key and all((output/p).is_file() and digest(output/p)==sha for p,sha in saved['checksums'].items()):
                accept(index,json.loads((output/'pairs'/pair_id/'metadata.json').read_text()),True);continue
        pending.append((index,task,pair_id,key))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(process,task,root,output,state,executable,args.threads,key,pair_id):index
                 for index,task,pair_id,key in pending}
        for future in as_completed(futures):
            try: accept(futures[future],future.result())
            except subprocess.CalledProcessError as e: raise RuntimeError(e.stderr) from e
    manifest={};summary={}
    for index,task in enumerate(tasks):
        m=results[index];row=[task['sourceScan'],task['targetScan'],f'{index:x}']
        if task['distanceRange']: row.append(task['distanceRange'])
        manifest.setdefault(task['dataset'],{}).setdefault(task['sequence'],[]).append(row)
        summary_row=summary.setdefault(task['dataset'],dict(pairs=0,successful=0))
        summary_row['pairs']+=1
        summary_row['successful']+=not m.get('error') and m['errors'][0]<=task['criteria'][0] and m['errors'][1]<=task['criteria'][1]
    write_json(output/'manifest.json',manifest)
    write_json(state/'summary.json',summary)
    write_json(state/'provenance.json',dict(config=config,sourceFiles=source_files,protocolFiles=protocols,inputFiles=input_hashes))
    print(f'Compact cache ready: {output}\nPrivate build state: {state}\n'+json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
