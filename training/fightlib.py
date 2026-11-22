"""Shared training code for the fight models (used by train_fight.py and tune_fight.py).

Works on the cached feature windows of the "features" stage (training/features.py), so an experiment never
recomputes features.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rover_ai.learned import FPS, PAIR_GAP
from rover_ai.nets import build_model, feature_groups

GROUPS = feature_groups()
VEL_LIKE = torch.from_numpy(GROUPS["velocity_like"])


def arch_config(P, kind=None):
    """params.yaml fight_model -> the architecture dict stored with the model (type + constructor arguments)."""
    kind = kind or P["arch"]
    return {"type": kind, **P[kind]}


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- augmentation
def augment(xb, aug, rng):
    """Training-time augmentation of a batch (B, T, F) float32 tensor (the mirror / person-swap variants are
    precomputed by the features stage)."""
    b, t, _ = xb.shape
    if aug.get("time_stretch", 0) > 0:
        # slow down: replay the last 75-100% of the window over the full 2 s (velocities shrink accordingly)
        pick = np.flatnonzero(rng.random(b) < aug["time_stretch"])
        for n in pick:
            length = int(rng.integers(int(0.75 * t), t))
            seg = xb[n, t - length:].T.unsqueeze(0)                           # (1, F, L)
            xb[n] = F.interpolate(seg, size=t, mode="linear", align_corners=True)[0].T
            xb[n, :, VEL_LIKE] *= (length - 1) / (t - 1)
    if aug.get("joint_dropout", 0) > 0:
        # a joint the pose model missed: position, visible flag and velocity are 0 for the whole window
        drop = torch.from_numpy(rng.random((b, 2, 17)) < aug["joint_dropout"])  # (B, person, joint)
        keep = (~drop).float()
        for g in ("pos", "vel"):
            idx = torch.from_numpy(GROUPS[g])                                  # (2, 17, 2)
            xb[:, :, idx] *= keep[:, None, :, :, None]
        xb[:, :, torch.from_numpy(GROUPS["vis"])] *= keep[:, None]
    if aug.get("noise", 0) > 0:
        xb = xb + aug["noise"] * torch.randn_like(xb)
    return xb


# ----------------------------------------------------------------------------- training
def train_model(P, arch, Xf, yf, Xd, yd, log=True, trial=None):
    """Fit a model on windows (Xf (N, 4, T, F) float16, yf); keep the epoch with the best dev window AUC.
    trial: an Optuna trial, reported every epoch so hopeless trials stop early."""
    model = build_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    pos_w = torch.tensor((1 - yf.mean()) / max(yf.mean(), 1e-6), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    aug = P.get("augment") or {}
    rng = np.random.default_rng(P.get("seed", 0))
    Xd_t = torch.from_numpy(Xd[:, 0].astype(np.float32))
    yf_t = torch.from_numpy(yf.astype(np.float32))
    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, P["epochs"] + 1):
        model.train()
        order = rng.permutation(len(yf))
        total = 0.0
        for i in range(0, len(order), P["batch"]):
            idx = order[i:i + P["batch"]]
            if len(idx) < 2:                                   # BatchNorm needs more than one sample
                continue
            v = rng.integers(0, 4, len(idx))                   # random geometric variant per sample
            xb = augment(torch.from_numpy(Xf[idx, v].astype(np.float32)), aug, rng)
            loss = loss_fn(model(xb), yf_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pd = torch.sigmoid(model(Xd_t)).numpy()
        auc = roc_auc(yd, pd)
        history.append(dict(epoch=epoch, loss=total / len(yf), dev_auc=auc))
        if log:
            print(f"epoch {epoch:2d}  loss {total / len(yf):.4f}  dev window AUC {auc:.4f}", flush=True)
        if trial is not None:
            trial.report(auc, epoch)
        if auc > best:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= P["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best


def windows_of(train, clips):
    """Training windows of the given clip indices."""
    X, y, clip = train
    sel = np.isin(clip, clips)
    return X[sel], y[sel]


# ----------------------------------------------------------------------------- clip-level scoring (like live)
def pair_probs(model, ev, clips, batch=2048):
    """Probability of every judged window of every close pair, for the given clips:
    {clip: {pair: (start frames, probabilities)}} (clips without a close pair are missing)."""
    X, clip, pair, start = ev
    sel = np.flatnonzero(np.isin(clip, clips))
    probs = np.empty(len(sel), np.float32)
    with torch.no_grad():
        for i in range(0, len(sel), batch):
            x = torch.from_numpy(X[sel[i:i + batch]].astype(np.float32))      # (b, 2, T, F): both person orders
            b = x.shape[0]
            p = torch.sigmoid(model(x.reshape(2 * b, *x.shape[2:]))).numpy().reshape(b, 2)
            probs[i:i + b] = p.mean(1)
    out = {}
    for c, pr, s, p in zip(clip[sel], pair[sel], start[sel], probs):
        out.setdefault(int(c), {}).setdefault(int(pr), ([], []))
        out[int(c)][int(pr)][0].append(int(s))
        out[int(c)][int(pr)][1].append(float(p))
    return {c: {k: (np.array(s), np.array(p)) for k, (s, p) in pairs.items()} for c, pairs in out.items()}


def clip_score(probs, need):
    """Live rule (rover_ai/learned.py): a pair alerts when its last `need` judgements are all above the threshold;
    its history survives detection gaps of up to PAIR_GAP seconds.
    Clip score = max over pairs and runs of (min probability over `need` judgements in a row); 0 without a pair."""
    best = 0.0
    for starts, p in (probs or {}).values():
        order = np.argsort(starts)
        run = []
        for s, v in zip(starts[order], p[order]):
            if run and s - run[-1][0] > PAIR_GAP * FPS:
                run = []
            run.append((s, v))
            if len(run) >= need:
                best = max(best, float(min(r[1] for r in run[-need:])))
    return best


def choose_rule(probs, y, need_choices):
    """Pick (judgements in a row, threshold) with the best accuracy on clips where a close pair was judged.
    Ties go to more judgements and a higher threshold (fewer false alarms)."""
    has = np.array([bool(p) for p in probs])
    grid = np.round(np.arange(0.30, 0.96, 0.01), 2)
    best = None
    for need in need_choices:
        scores = np.array([clip_score(p, need) for p in probs])
        for th in grid:
            key = (metrics(y[has], scores[has] >= th)["accuracy"], need, th)
            if best is None or key > best[0]:
                best = (key, scores)
    (_, need, th), scores = best
    return int(need), float(th), scores


# ----------------------------------------------------------------------------- metrics
def metrics(y, pred, score=None):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, tn = int((y & pred).sum()), int((~y & ~pred).sum())
    fp, fn = int((~y & pred).sum()), int((y & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    m = dict(accuracy=(tp + tn) / len(y) if len(y) else 0.0, precision=prec, recall=rec,
             f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0, tp=tp, tn=tn, fp=fp, fn=fn)
    if score is not None:
        m["roc_auc"] = roc_auc(y, score)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


def roc_auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties (Mann-Whitney U)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
