"""Trained fight model: a small network (GRU, TCN or ST-GCN, see nets.py) that reads the skeletons of two nearby people over 2 seconds.

The same feature code is used for training (training/train_fight.py) and live detection, so
the model sees exactly the same kind of input in both places.

Input window : T frames at 10 Hz of two people (17 COCO keypoints each, plus their boxes)
Features     : body-relative keypoint positions and velocities, normalised by body width,
               plus the distance between the two people and how fast it changes
Output       : probability that the pair is fighting
"""
import base64
import json
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np

from .events import Event
from .geometry import KP_CONF, TORSO, body_center, body_scale, box_iou

# No PyTorch import here: live detection with an ONNX or served model, and the stream workers, run without it.
# Every model below takes a numpy array (B, T, F) and returns numpy logits (B,).


def __getattr__(name):              # older imports: from rover_ai.learned import FightGRU, build_model
    if name in ("FightGRU", "build_model"):
        from . import nets
        return getattr(nets, name)
    raise AttributeError(name)

FPS = 10.0                      # the model works on 10 frames per second
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # left <-> right keypoints
N_FEAT = 2 * (17 * 3 + 17 * 2) + 4


def _person_frames_slow(kpts, boxes):
    """Reference version of _person_frames (per frame, with geometry.body_center / body_scale); used by tests."""
    centers = np.stack([body_center(k, b) for k, b in zip(kpts, boxes)])
    scale = float(np.median([body_scale(k, b) for k, b in zip(kpts, boxes)]))
    return centers, scale


_SH, _HP = [5, 6], [11, 12]                     # shoulders, hips (COCO)


def _masked_mean(pts, vis):
    """Mean of the visible points per frame: pts (T,n,2), vis (T,n) -> (T,2) (NaN where none is visible)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pts * vis[..., None]).sum(1) / vis.sum(1)[:, None]


def body_centers(kpts, boxes):
    """geometry.body_center for many poses at once: kpts (N,17,3), boxes (N,4) -> (N,2)."""
    k, b = np.asarray(kpts), np.asarray(boxes)
    torso = k[:, TORSO, 2] > KP_CONF
    box_c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    return np.where(torso.any(1)[:, None], _masked_mean(k[:, TORSO, :2], torso), box_c)


def body_scales(kpts, boxes):
    """geometry.body_scale (body width in pixels) for many poses at once: kpts (N,17,3), boxes (N,4) -> (N,)."""
    k, b = np.asarray(kpts), np.asarray(boxes)
    vis = k[:, :, 2] > KP_CONF
    both = vis[:, 5] & vis[:, 6]
    width = np.where(both, np.linalg.norm(k[:, 5, :2] - k[:, 6, :2], axis=1), -np.inf)
    sh, hp = vis[:, _SH], vis[:, _HP]
    trunk = np.where(sh.any(1) & hp.any(1),
                     0.75 * np.linalg.norm(_masked_mean(k[:, _SH, :2], sh) - _masked_mean(k[:, _HP, :2], hp), axis=1),
                     -np.inf)
    best = np.maximum(width, trunk)
    return np.where(np.isfinite(best), np.maximum(best, 1.0), np.maximum((b[:, 3] - b[:, 1]) / 4, 1.0))


def _person_frames(kpts, boxes):
    """Per-frame body centre (T,2) and one body width for the whole window (median over the frames)."""
    return body_centers(kpts, boxes), float(np.median(body_scales(kpts, boxes)))


def window_features(ka, ba, kb, bb):
    """ka, kb: (T,17,3) keypoints; ba, bb: (T,4) boxes of the two people -> (T, N_FEAT) float32."""
    ca, sa = _person_frames(ka, ba)
    cb, sb = _person_frames(kb, bb)
    s = max((sa + sb) / 2, 1.0)
    mid = (ca + cb) / 2                                     # pair centre per frame
    feats = []
    for k, c in ((ka, ca), (kb, cb)):
        vis = (k[:, :, 2] > KP_CONF).astype(np.float32)     # (T,17)
        pos = (k[:, :, :2] - mid[:, None]) / s * vis[..., None]
        own = (k[:, :, :2] - c[:, None]) / s                # relative to the person's own body
        vel = np.zeros_like(own)
        vel[1:] = (own[1:] - own[:-1]) * FPS
        vel[1:] *= (vis[1:] * vis[:-1])[..., None]
        feats += [pos.reshape(len(k), -1), vis, vel.reshape(len(k), -1)]
    d = (cb - ca) / s
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    ddist = np.zeros_like(dist)
    ddist[1:] = (dist[1:] - dist[:-1]) * FPS
    feats += [d, dist, ddist]
    return np.clip(np.concatenate(feats, axis=1), -10, 10).astype(np.float32)


def flip_kpts(k, width):
    """Mirror keypoints horizontally (training augmentation)."""
    k = k[:, FLIP_IDX].copy()
    k[..., 0] = width - k[..., 0]
    return k


def flip_boxes(b, width):
    b = b.copy()
    b[:, [0, 2]] = width - b[:, [2, 0]]
    return b


class OnnxModel:
    """An exported model run with ONNX Runtime (no PyTorch needed, e.g. on the Pi or in the serving container).
    Called like the PyTorch model: tensor (B, T, F) -> logits tensor (B,)."""

    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        return self.session.run(None, {self.input: x.astype(np.float32)})[0].reshape(-1)

    def eval(self):
        return self


class RemoteModel:
    """The model behind the serving API (serving/app.py), called like the local one: (B, T, F) -> logits (B,).
    url = http://<host>:8000/v1/models/<activity>. Its settings (window, threshold, pairing rules) come from the
    server, and are refreshed when the server hot-reloads another registry version (on_change callbacks)."""

    def __init__(self, url, timeout=2.0):
        from urllib.parse import urlsplit
        self.url = url.rstrip("/")
        parts = urlsplit(self.url)
        self.https, self.host, self.path = parts.scheme == "https", parts.netloc, parts.path
        self.timeout = timeout
        self.conn = None                    # one kept-alive connection (a new TCP connection per frame is slow)
        self.on_change = []
        self.meta = self._fetch_meta()

    def _request(self, path, body=None):
        import http.client
        data = None if body is None else json.dumps(body).encode()
        for attempt in (1, 2):              # the server may have closed an idle kept-alive connection
            if self.conn is None:
                cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
                self.conn = cls(self.host, timeout=self.timeout)
            try:
                if self.conn.sock is None:
                    import socket
                    self.conn.connect()
                    self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # no Nagle delay
                self.conn.request("GET" if data is None else "POST", path, body=data,
                                  headers={"Content-Type": "application/json"})
                r = self.conn.getresponse()
                payload = r.read()
            except (OSError, http.client.HTTPException) as e:
                self.conn.close()
                self.conn = None
                if attempt == 2:
                    raise OSError(f"serving API: {e}") from e
                continue
            if r.status != 200:
                raise OSError(f"serving API {path}: HTTP {r.status} {payload[:200]!r}")
            return json.loads(payload)

    def _fetch_meta(self):
        info = self._request(self.path)
        return dict(window=info["window"], threshold=info["threshold"], consecutive=info.get("consecutive"),
                    proximity=info.get("proximity"), min_cover=info.get("min_cover"), arch={"type": info["arch"]},
                    registry=dict(model=info.get("registered_model"), version=info.get("version"),
                                  alias="served", run_id=info.get("run_id")), served_by=self.url)

    def __call__(self, x):
        x = np.ascontiguousarray(x, "<f4")                  # compact binary payload (base64 float32)
        out = self._request(self.path + "/predict",
                            {"shape": list(x.shape), "data_b64": base64.b64encode(x.tobytes()).decode()})
        if str(out.get("model_version")) != str(self.meta["registry"]["version"]):     # server reloaded a model
            self.meta = self._fetch_meta()
            for callback in self.on_change:
                callback(self.meta)
        return np.asarray(out["logits"], np.float32)

    def eval(self):
        return self


def is_remote(ref):
    return str(ref).startswith(("http://", "https://"))


def load_model(path):
    """models/<name>.pt (PyTorch weights) or models/<name>.onnx, with the settings in models/<name>.json,
    or the URL of a model on the serving API."""
    if is_remote(path):
        model = RemoteModel(str(path))
        return model, model.meta
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if path.suffix == ".onnx":
        return OnnxModel(path), meta
    return TorchModel(path, meta), meta


class TorchModel:
    """PyTorch weights (models/<name>.pt), with the same numpy interface as the ONNX model."""

    def __init__(self, path, meta):
        import torch

        from .nets import build_model
        self.torch = torch
        self.net = build_model(meta["arch"])
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()

    def __call__(self, x):
        with self.torch.no_grad():
            return self.net(self.torch.from_numpy(np.asarray(x, np.float32))).numpy().reshape(-1)

    def eval(self):
        return self


def pair_probability(model, ka, ba, kb, bb):
    """Average over both person orders, so the answer doesn't depend on who is 'A'."""
    return pair_probabilities(model, [(ka, ba, kb, bb)])[0]


def pair_probabilities(model, pairs):
    """Several pairs [(ka, ba, kb, bb), ...] in one model call (one request when the model is remote)."""
    x = np.stack([w for ka, ba, kb, bb in pairs
                  for w in (window_features(ka, ba, kb, bb), window_features(kb, bb, ka, ba))])
    logits = np.asarray(model(x.astype(np.float32)), np.float64).reshape(-1, 2)
    return [float(v) for v in (1.0 / (1.0 + np.exp(-logits))).mean(1)]


MIN_COVER = 0.5                 # a person must have been tracked for at least half the window (1 s)
PAIR_GAP = 1.0                  # a pair keeps its alert history through detection gaps of up to 1 s


def resample_track(hist, t_end, T, min_cover=MIN_COVER):
    """Track history -> (T,17,3) keypoints and (T,4) boxes at 10 Hz ending at t_end (linear interpolation;
    before the track started, its first pose is repeated). None if tracked for less than min_cover of the window."""
    times = np.array([h[0] for h in hist])
    t_query = t_end - (T - 1 - np.arange(T)) / FPS
    if (t_query >= times[0] - 0.05).mean() < min_cover:
        return None
    k = np.stack([h[1] for h in hist])
    b = np.stack([h[2] for h in hist])
    idx = np.clip(np.searchsorted(times, t_query), 1, len(times) - 1) if len(times) > 1 else np.zeros(T, int)
    if len(times) == 1:
        return np.repeat(k, T, 0), np.repeat(b, T, 0)
    t0, t1 = times[idx - 1], times[idx]
    w = np.clip((t_query - t0) / np.maximum(t1 - t0, 1e-6), 0, 1)
    kk = k[idx - 1] * (1 - w)[:, None, None] + k[idx] * w[:, None, None]
    bb = b[idx - 1] * (1 - w)[:, None] + b[idx] * w[:, None]
    return kk, bb


class LearnedFightDetector:
    """Drop-in replacement for the rule-based FightDetector, driven by the trained model."""

    def __init__(self, cfg, root):
        ref = cfg["model"] if is_remote(cfg["model"]) else Path(root) / cfg["model"]
        self.cfg = cfg
        self.model, meta = load_model(ref)
        self.apply_meta(meta)
        if isinstance(self.model, RemoteModel):          # the server switched to another registry version
            self.model.on_change.append(self.apply_meta)
        self.window = cfg["window_seconds"]
        self.failed_at = None                            # serving API unreachable since (remote model)
        self.pairs = {}                 # pair -> last `need` decisions (above threshold or not)
        self.last = {}                  # pair -> time it was last judged
        self.probs = {}

    def apply_meta(self, meta):
        cfg = self.cfg
        self.meta = meta
        self.T = meta["window"]
        self.thr = cfg.get("model_threshold") or meta["threshold"]
        # pairing rules the model was trained with (stored in its .json); config.yaml only for older models
        self.proximity = meta.get("proximity") or cfg["proximity"]
        self.min_cover = meta.get("min_cover") or MIN_COVER
        self.need = cfg.get("model_consecutive") or meta.get("consecutive") or 3

    def get_state(self):
        """Per-pair alert history as plain data (the stream keeps it between micro-batches)."""
        return {"pairs": {f"{a}-{b}": list(h) for (a, b), h in self.pairs.items()},
                "last": {f"{a}-{b}": v for (a, b), v in self.last.items()}}

    def set_state(self, state):
        def key(k):
            a, b = k.split("-")
            return int(a), int(b)
        state = state or {}
        self.pairs = {key(k): deque(v, maxlen=self.need) for k, v in state.get("pairs", {}).items()}
        self.last = {key(k): v for k, v in state.get("last", {}).items()}
        self.probs = {}

    def judge(self, windows, t):
        """Probabilities of all candidate pairs of this frame (one model call), or None if the serving API
        can't be reached (no judgements; warned every 10 s)."""
        try:
            probs = pair_probabilities(self.model, windows)
        except OSError as e:                             # connection refused, timeout, HTTP error
            if self.failed_at is None or t - self.failed_at > 10:
                print(f"Fight model: {e}; no fight judgements until it is back")
                self.failed_at = t
            return None
        self.failed_at = None
        return probs

    def update(self, tracks, t, flags):
        events = []
        energy = {tr.id: tr.motion_energy(self.window) for tr in tracks}
        candidates, windows = [], []
        for a, b in combinations(sorted(tracks, key=lambda tr: tr.id), 2):
            s = (a.scale + b.scale) / 2
            close = np.linalg.norm(a.center - b.center) < self.proximity * s or box_iou(a.box, b.box) > 0.05
            if not close:
                continue
            wa, wb = resample_track(a.hist, t, self.T, self.min_cover), resample_track(b.hist, t, self.T, self.min_cover)
            if wa is None or wb is None:
                continue
            candidates.append((a, b))
            windows.append((*wa, *wb))
        probs = self.judge(windows, t) if windows else []
        seen = set()
        for (a, b), p in zip(candidates, probs or []):
            key = (a.id, b.id)
            seen.add(key)
            self.probs[key] = p
            if t - self.last.get(key, t) > PAIR_GAP:      # lost for too long: start counting again
                self.pairs.pop(key, None)
            self.last[key] = t
            hist = self.pairs.setdefault(key, deque(maxlen=self.need))
            hist.append(p >= self.thr)
            if len(hist) == self.need and all(hist):
                flags[a.id].add("FIGHT")
                flags[b.id].add("FIGHT")
                events.append(Event("fight", f"fight-{a.id}-{b.id}",
                                    f"FIGHT: ID {a.id} and ID {b.id} ({p:.0%})", [a.id, b.id]))
        for key in [k for k in self.probs if k not in seen]:
            self.probs.pop(key, None)
        for key in [k for k, last in self.last.items() if t - last > PAIR_GAP]:
            self.pairs.pop(key, None)                    # keep a pair's history through short detection gaps
            self.last.pop(key, None)
        return events, energy
