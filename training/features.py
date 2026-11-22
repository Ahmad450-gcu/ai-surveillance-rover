"""Pipeline stage "features": validated skeleton clips -> model-ready feature windows (processed zone).

    python training/features.py              # settings: params.yaml sections "features" and "fight_model"
    python training/features.py --upload     # also copy the result to s3://processed/

Computing features is the slow part of training (about 9 minutes on a laptop), so it is done once here and
every experiment (cross-validation, tuning, every architecture) reuses it. It reruns only when the data or
the feature settings change (DVC).

Output  data/processed/fight/<dataset>/
    train.npz   training windows of the train split, with the 4 precomputed variants
                (as is, mirrored, persons swapped, both): X (N, 4, T, F) float16, y (N,), clip (N,)
    eval.npz    every window (stride 1, both person orders) of every close pair in every clip, used to score whole
                clips exactly like live detection: X (M, 2, T, F) float16, clip (M,), pair (M,), start (M,)
    index.json  the clips (train first, then test), labels and the settings used
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rover_ai.geometry import body_center, body_scale, box_iou  # noqa: E402
from rover_ai.learned import MIN_COVER, N_FEAT, flip_boxes, flip_kpts, window_features  # noqa: E402

FEATURE_VERSION = 1             # bump when window_features changes, so old caches are not mixed with new code


def settings():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    return params["features"]


# ----------------------------------------------------------------------------- clips -> tracks -> pairs
def load_clip(path):
    """npz -> {track_id: (kpts (F,17,3), boxes (F,4), alive (F,), present (F,))}, n_frames, width"""
    d = np.load(path)
    F = int(d["n_frames"])
    tracks = {}
    for tid in np.unique(d["track"]):
        sel = d["track"] == tid
        fr = d["frame"][sel].astype(int)
        if len(fr) < 3:
            continue
        k = np.zeros((F, 17, 3), np.float32)
        b = np.zeros((F, 4), np.float32)
        present = np.zeros(F, bool)
        k[fr], b[fr], present[fr] = d["kpts"][sel], d["box"][sel], True
        idx = np.flatnonzero(present)
        for i0, i1 in zip(idx[:-1], idx[1:]):           # fill gaps by interpolation, like live
            if i1 - i0 > 1:
                w = (np.arange(i0 + 1, i1) - i0) / (i1 - i0)
                k[i0 + 1:i1] = k[i0] * (1 - w)[:, None, None] + k[i1] * w[:, None, None]
                b[i0 + 1:i1] = b[i0] * (1 - w)[:, None] + b[i1] * w[:, None]
        first = idx[0]
        k[:first], b[:first] = k[first], b[first]       # before the track started: repeat its first pose, like live
        alive = np.zeros(F, bool)
        alive[first:idx[-1] + 1] = True                 # tracked from first to last sighting
        tracks[int(tid)] = (k, b, alive, present)
    return tracks, F, float(d["width"])


def close_pairs(tracks, s, e, proximity, min_cover=MIN_COVER):
    """Pairs of people seen in the window's last frame, tracked for at least min_cover of the window,
    and close to each other (judged at the last frame, like live), closest first."""
    ids = [i for i, (_, _, alive, pr) in tracks.items()
           if pr[e - 1] and alive[s:e].mean() >= min_cover and alive[e - 1]]
    out = []
    m = e - 1
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            ka, ba = tracks[ids[x]][:2]
            kb, bb = tracks[ids[y]][:2]
            sc = (body_scale(ka[m], ba[m]) + body_scale(kb[m], bb[m])) / 2
            dist = np.linalg.norm(body_center(ka[m], ba[m]) - body_center(kb[m], bb[m])) / sc
            if dist < proximity or box_iou(ba[m], bb[m]) > 0.05:
                out.append((dist, ids[x], ids[y]))
    return [(a, b) for _, a, b in sorted(out)]


def variants(ka, ba, kb, bb, width):
    """Feature windows for the 4 geometric augmentations: as is, mirrored, persons swapped, both."""
    fa, fba, fb, fbb = flip_kpts(ka, width), flip_boxes(ba, width), flip_kpts(kb, width), flip_boxes(bb, width)
    return np.stack([window_features(ka, ba, kb, bb), window_features(fa, fba, fb, fbb),
                     window_features(kb, bb, ka, ba), window_features(fb, fbb, fa, fba)])


def clip_features(path, label, train, cfg):
    """One clip -> (training windows (n, 4, T, F) or None, eval windows [(pair, start, AB, BA)])."""
    T, stride, prox, cover = cfg["window"], cfg["stride"], cfg["proximity"], cfg.get("min_cover", MIN_COVER)
    tracks, F, width = load_clip(path)
    train_x = []
    eval_rows = []
    pair_ids = {}
    for s in range(0, F - T + 1):
        e = s + T
        pairs = close_pairs(tracks, s, e, prox, cover)
        if train and s % stride == 0:
            # fight clip: the closest pair is (almost always) the fighters; normal clip: every close pair is a negative
            for a, b in (pairs[:1] if label else pairs):
                ka, ba = tracks[a][:2]
                kb, bb = tracks[b][:2]
                train_x.append(variants(ka[s:e], ba[s:e], kb[s:e], bb[s:e], width))
        for a, b in pairs:
            ka, ba = tracks[a][:2]
            kb, bb = tracks[b][:2]
            pid = pair_ids.setdefault((a, b), len(pair_ids))
            eval_rows.append((pid, s, window_features(ka[s:e], ba[s:e], kb[s:e], bb[s:e]),
                              window_features(kb[s:e], bb[s:e], ka[s:e], ba[s:e])))
    tx = np.array(train_x, np.float16).reshape(-1, 4, T, N_FEAT) if train else None
    return tx, eval_rows


def _work(job):
    path, label, train, cfg = job
    tx, rows = clip_features(path, label, train, cfg)
    if rows:
        ex = np.stack([np.stack([r[2], r[3]]) for r in rows]).astype(np.float16)
        meta = np.array([(r[0], r[1]) for r in rows], np.int32)
    else:
        ex, meta = np.zeros((0, 2, cfg["window"], N_FEAT), np.float16), np.zeros((0, 2), np.int32)
    return tx, ex, meta


def listing(roots, split):
    return sorted((p.relative_to(ROOT).as_posix(), int(p.parent.name == "Fight"))
                  for r in roots for p in (ROOT / r / split).glob("*/*.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="copy the features to the lake (s3://processed/)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    cfg = settings()
    roots = [cfg["data"]] + list(cfg.get("extra_data") or [])
    train, test = listing(roots, "train"), listing(roots, "val")
    clips = train + test
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(str(ROOT / p), lab, i < len(train), cfg) for i, (p, lab) in enumerate(clips)]
    tr_x, tr_y, tr_clip, ev_x, ev_meta = [], [], [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (tx, ex, meta) in enumerate(pool.map(_work, jobs, chunksize=8)):
            if tx is not None and len(tx):
                tr_x.append(tx); tr_y.append(np.full(len(tx), clips[i][1], np.int8))
                tr_clip.append(np.full(len(tx), i, np.int32))
            if len(ex):
                ev_x.append(ex); ev_meta.append(np.column_stack([np.full(len(ex), i, np.int32), meta]))
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(clips)} clips  {time.time() - t0:.0f}s", flush=True)
    ev_meta = np.concatenate(ev_meta)
    np.savez(out / "train.npz", X=np.concatenate(tr_x), y=np.concatenate(tr_y), clip=np.concatenate(tr_clip))
    np.savez(out / "eval.npz", X=np.concatenate(ev_x), clip=ev_meta[:, 0], pair=ev_meta[:, 1],
             start=ev_meta[:, 2].astype(np.int16))
    n_tr, n_ev = sum(map(len, tr_x)), len(ev_meta)
    index = dict(feature_version=FEATURE_VERSION, n_feat=N_FEAT, settings=cfg, n_train=len(train), n_test=len(test),
                 clips=[dict(path=p, label=lab, split="train" if i < len(train) else "test")
                        for i, (p, lab) in enumerate(clips)],
                 train_windows=n_tr, eval_windows=n_ev,
                 clips_with_close_pair=int(len(np.unique(ev_meta[:, 0]))))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"features: {n_tr} training windows, {n_ev} eval windows, {index['clips_with_close_pair']}/{len(clips)} "
          f"clips with a close pair | {(time.time() - t0) / 60:.1f} min -> {out.relative_to(ROOT)}")
    if args.upload:
        from mlops import lake
        prefix = "fight/" + out.name
        up, same = lake.sync_dir(out, "processed", prefix, "*")
        print(f"lake: s3://processed/{prefix}/  {up} uploaded, {same} unchanged")


def load_features(cfg=None):
    """-> dict(index, train=(X, y, clip), eval=(X, clip, pair, start)) from the processed zone."""
    cfg = cfg or settings()
    out = ROOT / cfg["output"]
    index = json.loads((out / "index.json").read_text())
    if index.get("feature_version") != FEATURE_VERSION:
        sys.exit(f"{out} was built by another feature version: run  dvc repro features")
    tr, ev = np.load(out / "train.npz"), np.load(out / "eval.npz")
    return dict(index=index, train=(tr["X"], tr["y"], tr["clip"]),
                eval=(ev["X"], ev["clip"], ev["pair"], ev["start"].astype(np.int32)))


if __name__ == "__main__":
    main()
