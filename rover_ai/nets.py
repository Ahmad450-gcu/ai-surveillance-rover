"""Fight model architectures. Every model reads the same input, a (B, T, N_FEAT) window of two people
(rover_ai/learned.py: window_features), and returns one logit per window, so live detection, training,
ONNX export and serving work the same for all of them.

    gru     recurrent baseline: per-frame MLP -> 2-layer GRU -> last state + mean over time
    tcn     temporal convolutional network: dilated 1-D convolutions over time (parallel, fast on CPU)
    stgcn   spatial-temporal graph convolution (ST-GCN, Yan et al. 2018): the 34 joints of the two people
            are graph nodes; edges follow the skeleton plus person-to-person contact edges

Feature layout (N_FEAT = 174), per person p in {A, B} at offset o = 85 * p:
    o +  0 .. 34   joint positions relative to the pair centre (x, y), / body width
    o + 34 .. 51   joint visible (0/1)
    o + 51 .. 85   joint velocities relative to the person's own body (x, y)
    170, 171       vector from A to B      172  distance      173  change of distance
"""
import numpy as np
import torch
from torch import nn

N_JOINTS = 17
PERSON = 85                     # features per person
PAIR_FEATS = slice(170, 174)
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
STRIKERS = (9, 10, 15, 16)      # wrists and ankles: what hits
TARGETS = (0, 5, 6, 11, 12)     # head, shoulders, hips: what gets hit


def feature_groups(n_feat=174):
    """Index arrays of the feature groups (used by augmentation)."""
    pos, vis, vel = [], [], []
    for p in range(2):
        o = p * PERSON
        pos.append(np.arange(o, o + 34).reshape(17, 2))
        vis.append(np.arange(o + 34, o + 51))
        vel.append(np.arange(o + 51, o + 85).reshape(17, 2))
    return dict(pos=np.stack(pos), vis=np.stack(vis), vel=np.stack(vel), velocity_like=np.concatenate(
        [np.stack(vel).ravel(), [173]]))


class FightGRU(nn.Module):
    def __init__(self, n_feat=174, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):                       # x: (B, T, F)
        h, _ = self.gru(self.inp(x))
        z = torch.cat([h[:, -1], h.mean(1)], dim=1)   # last state + average over time
        return self.head(z).squeeze(1)          # logits


class _TemporalBlock(nn.Module):
    def __init__(self, c, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(c, c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c, c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(c))

    def forward(self, x):
        return torch.relu(x + self.net(x))


class FightTCN(nn.Module):
    def __init__(self, n_feat=174, channels=64, levels=3, kernel=3, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(n_feat, channels, 1), nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[_TemporalBlock(channels, kernel, 2 ** i, dropout) for i in range(levels)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels, channels), nn.ReLU(), nn.Linear(channels, 1))

    def forward(self, x):                       # x: (B, T, F)
        h = self.blocks(self.inp(x.transpose(1, 2)))   # (B, C, T)
        z = torch.cat([h[:, :, -1], h.mean(2)], dim=1)
        return self.head(z).squeeze(1)


def pair_adjacency():
    """Normalised adjacency (34 x 34) of two skeletons plus contact edges between them."""
    n = 2 * N_JOINTS
    a = np.eye(n, dtype=np.float32)
    for p in range(2):
        for i, j in COCO_EDGES:
            a[p * 17 + i, p * 17 + j] = a[p * 17 + j, p * 17 + i] = 1
    for j in range(N_JOINTS):                                   # same joint of both people
        a[j, 17 + j] = a[17 + j, j] = 1
    for s in STRIKERS:                                          # A's hands/feet <-> B's head/torso and back
        for t in TARGETS:
            a[s, 17 + t] = a[17 + t, s] = 1
            a[17 + s, t] = a[t, 17 + s] = 1
    d = 1 / np.sqrt(a.sum(1))
    return (a * d[:, None] * d[None]).astype(np.float32)


class _STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dropout):
        super().__init__()
        self.gcn = nn.Conv2d(c_in, c_out, 1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.tcn = nn.Sequential(nn.Conv2d(c_out, c_out, (kernel, 1), padding=(kernel // 2, 0)), nn.BatchNorm2d(c_out),
                                 nn.Dropout(dropout))
        self.res = nn.Identity() if c_in == c_out else nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.BatchNorm2d(c_out))

    def forward(self, x, adj):                  # x: (B, C, T, V), adj: (V, V)
        h = torch.relu(self.bn1(self.gcn(torch.matmul(x, adj))))
        return torch.relu(self.tcn(h) + self.res(x))


class FightSTGCN(nn.Module):
    def __init__(self, n_feat=174, channels=32, blocks=3, kernel=5, dropout=0.2):
        super().__init__()
        g = feature_groups()
        # per joint: position (2) + visible (1) + velocity (2) = 5 channels, 34 joints (both people)
        idx = np.stack([g["pos"][..., 0], g["pos"][..., 1], g["vis"], g["vel"][..., 0], g["vel"][..., 1]])  # (5,2,17)
        self.register_buffer("idx", torch.from_numpy(idx.reshape(5, 34)).long(), persistent=False)
        self.register_buffer("adj", torch.from_numpy(pair_adjacency()), persistent=False)
        self.adaptive = nn.Parameter(torch.zeros(34, 34))      # learned extra edges (2s-AGCN style)
        self.bn_in = nn.BatchNorm1d(5 * 34)
        layers, c = [], 5
        for i in range(blocks):
            c_out = channels * (2 if i == blocks - 1 and blocks > 1 else 1)
            layers.append(_STGCNBlock(c, c_out, kernel, dropout))
            c = c_out
        self.blocks = nn.ModuleList(layers)
        self.pair = nn.Sequential(nn.Conv1d(4, 16, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c + 32, c), nn.ReLU(), nn.Linear(c, 1))

    def forward(self, x):                       # x: (B, T, F)
        b, t, _ = x.shape
        j = x[:, :, self.idx]                   # (B, T, 5, 34)
        j = self.bn_in(j.reshape(b, t, 5 * 34).transpose(1, 2)).reshape(b, 5, 34, t).transpose(2, 3)  # (B, 5, T, 34)
        adj = self.adj + self.adaptive
        for blk in self.blocks:
            j = blk(j, adj)
        zg = j.mean(dim=(2, 3))                 # (B, C)
        p = self.pair(x[:, :, PAIR_FEATS].transpose(1, 2))   # (B, 16, T)
        z = torch.cat([zg, p.mean(2), p[:, :, -1]], dim=1)
        return self.head(z).squeeze(1)


ARCHS = {"gru": FightGRU, "tcn": FightTCN, "stgcn": FightSTGCN}


def build_model(arch):
    """arch: dict with "type" (gru | tcn | stgcn, default gru for older models) and the constructor arguments."""
    arch = dict(arch)
    kind = arch.pop("type", "gru")
    if kind not in ARCHS:
        raise ValueError(f"unknown fight model type {kind!r}, choose from {sorted(ARCHS)}")
    return ARCHS[kind](**arch)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
