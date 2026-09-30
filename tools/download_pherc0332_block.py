"""(025, 2026-09-27: copy with START/END = 3703..4744, i.e. 1 cm axial centred on Z4224)
Download the exact L2 Z3703..4744 block, preserving source uint8 voxels.

Public uncompressed Zarr v2 chunks permit HTTP byte ranges along Z.
Only verified 404 responses represent Zarr fill; other errors stop completion.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

import numpy as np

BASE = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr'
START, END = 3703, 4744


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def get_json(suffix):
    with urllib.request.urlopen(BASE + suffix, timeout=45) as response:
        return json.load(response)


def fetch(job):
    cz, cy, cx = job
    lo, hi = max(START, cz*128), min(END+1, (cz+1)*128)
    first = (lo-cz*128)*128*128
    last = (hi-cz*128)*128*128-1
    url = f'{BASE}/2/{cz}/{cy}/{cx}'
    request = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}',
                                                   'Accept-Encoding': 'identity'})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise ValueError(f'Expected HTTP 206, got {response.status}: {url}')
                expected_range = f'bytes {first}-{last}/{128**3}'
                if response.headers.get('Content-Range') != expected_range:
                    raise ValueError(f'Unexpected Content-Range: {response.headers.get("Content-Range")}')
                data = response.read(last-first+2)
                if len(data) != last-first+1:
                    raise ValueError('Truncated or oversized range response')
                return job, data, {'status': 206, 'etag': response.headers.get('ETag'),
                                  'range': expected_range, 'response_sha256': digest(data)}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return job, None, {'status': 404, 'interpretation': 'Zarr absent chunk: fill_value=0'}
            if attempt == 5:
                raise
        except Exception:
            if attempt == 5:
                raise
        time.sleep(min(2**attempt, 16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    meta, attrs = get_json('/2/.zarray'), get_json('/.zattrs')
    expected = {'shape': [8398,3941,3941], 'chunks': [128,128,128], 'dtype': '|u1',
                'fill_value': 0, 'order': 'C', 'compressor': None, 'filters': None,
                'dimension_separator': '/', 'zarr_format': 2}
    if any(meta.get(k) != v for k,v in expected.items()):
        raise ValueError(f'Source metadata differs from downloader assumptions: {meta}')
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / f'PHerc0332_L2_Z{START}-{END}_uint8.npy'
    partial = args.out / 'volume.partial.npy'
    journal_path = args.out / 'chunks.json'
    if final.exists():
        raise FileExistsError(f'Completed file already exists: {final}')
    shape = (END-START+1, 3941, 3941)
    if not partial.exists() and shutil.disk_usage(args.out).free < np.prod(shape)+1024**3:
        raise OSError('Insufficient free disk space')
    for name, obj in [('source_zarray.json',meta), ('source_zattrs.json',attrs)]:
        path = args.out/name
        if path.exists() and json.loads(path.read_text()) != obj:
            raise ValueError('Source metadata changed since previous download')
        path.write_text(json.dumps(obj,indent=2))
    completed = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    if partial.exists():
        volume = np.lib.format.open_memmap(partial, mode='r+')
        if volume.shape != shape or volume.dtype != np.uint8:
            raise ValueError('Partial volume has wrong shape or dtype')
    else:
        if completed:
            raise ValueError('Journal exists without partial volume')
        volume = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8, shape=shape)
        volume[:] = 0
        volume.flush()
    def region(job):
        cz,cy,cx = job
        lo,hi = max(START,cz*128), min(END+1,(cz+1)*128)
        return (slice(lo-START,hi-START),slice(cy*128,min((cy+1)*128,3941)),
                slice(cx*128,min((cx+1)*128,3941)))
    jobs = [(cz,cy,cx) for cz in range(START//128,END//128+1) for cy in range(31) for cx in range(31)]
    for job in jobs:
        key = '/'.join(map(str,job))
        if key in completed and digest(volume[region(job)].tobytes()) != completed[key]['stored_sha256']:
            del completed[key]
    pending = [job for job in jobs if '/'.join(map(str,job)) not in completed]
    initial_count = len(completed)
    print(f'Downloading {len(pending)} chunks; {initial_count} verified from resume; shape={shape}', flush=True)
    def checkpoint():
        volume.flush()
        tmp = journal_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(completed,indent=2))
        tmp.replace(journal_path)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch,job) for job in pending]
        for future in as_completed(futures):
            job,data,record = future.result()
            slices = region(job)
            target = volume[slices]
            if data is not None:
                slab = np.frombuffer(data,dtype=np.uint8).reshape(target.shape[0],128,128)
                target[:] = slab[:,:target.shape[1],:target.shape[2]]
            else:
                target[:] = 0
            record['stored_sha256'] = digest(target.tobytes())
            record['download_bytes'] = len(data) if data is not None else 0
            completed['/'.join(map(str,job))] = record
            if (len(completed)-initial_count) % 32 == 0:
                checkpoint()
                print(f'{len(completed)}/{len(jobs)} chunks verified; {time.monotonic()-started:.0f}s',flush=True)
    checkpoint()
    if len(completed) != len(jobs):
        raise ValueError('Incomplete download')
    for job in jobs:
        key = '/'.join(map(str,job))
        if digest(volume[region(job)].tobytes()) != completed[key]['stored_sha256']:
            raise ValueError(f'Readback checksum mismatch: {key}')
    reference = np.load(args.reference,allow_pickle=False)
    if not np.array_equal(volume[4224-START],reference):
        raise ValueError('Central slice differs from the previously used scan; inspect before completion')
    slice_hashes = [{'z_L2':START+i, 'index_in_block':i,
                     'sha256_raw_uint8_c_order':digest(volume[i].tobytes())} for i in range(shape[0])]
    del target, volume
    partial.replace(final)
    manifest = {'status':'complete', 'scan_id':'PHerc0332/20251211183505', 'level':2,
                'source_url':BASE, 'z_first_inclusive':START, 'z_last_inclusive':END,
                'shape_zyx':list(shape), 'dtype':'uint8', 'spacing_um_zyx':[9.596]*3,
                'spacing_provenance':'source name 2.399um times multiscale level-2 factor 4',
                'origin_source_L2_zyx':[START,0,0], 'center_index':4224-START,
                'voxel_values':'unchanged source values; no resampling or conversion',
                'file':final.name, 'file_bytes':final.stat().st_size, 'file_sha256':file_hash(final),
                'chunks_total':len(jobs), 'chunks_absent_fill_zero':sum(r['status']==404 for r in completed.values()),
                'downloaded_payload_bytes':sum(r['download_bytes'] for r in completed.values()),
                'central_slice_matches_previous_reference':True, 'reference_sha256':file_hash(args.reference),
                'slice_hashes':slice_hashes}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'README.md').write_text(
        f'# PHerc0332 — L2, Z={START} à {END} inclus\n\n'
        f'{END-START+1} coupes, grille ZYX = {END-START+1} × 3941 × 3941, uint8 natif, pas 9,596 µm.\n'
        'Les valeurs du volume public sont conservées sans conversion ni rééchantillonnage.\n'
        'Le niveau L2 est un niveau réduit du scan natif à 2,399 µm.\n'
        'La coupe centrale correspond exactement à la coupe locale utilisée pour le guidage.\n\n'
        '```python\nimport numpy as np\n'
        f"volume = np.load('{final.name}', mmap_mode='r')\n"
        'coupe_centrale = volume[100]  # Z=4224 au niveau L2\n'
        f'coupe_z = volume[z - {START}]    # {START} <= z <= {END}\n```\n\n'
        'manifest.json contient les empreintes, la calibration et les indices.\n'
        'chunks.json conserve la provenance HTTP et les contrôles des blocs téléchargés.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='slice_hashes'},indent=2),flush=True)


if __name__ == '__main__':
    main()
