"""Alert handling: cooldown, snapshot + pre-event clip, JSON log, sound, and listeners (e.g. the backend)."""
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

SOUND = "/System/Library/Sounds/Sosumi.aiff"


class AlertManager:
    def __init__(self, cfg, root: Path):
        self.dir = root / cfg["dir"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self.dir / "alerts.jsonl"
        self.cooldown = cfg["cooldown_seconds"]
        self.type_cooldown = cfg.get("type_cooldown_seconds", 0)
        self.last_type = {}
        self.clip_seconds = cfg["clip_seconds"]
        self.sound = cfg["sound"] and sys.platform == "darwin"
        self.listeners = []                 # called with each alert record (dict)
        self.last = {}
        self.recent = deque(maxlen=5)       # (wall time, message) for the on-screen banner
        self.buffer = deque(maxlen=int(self.clip_seconds * 30) + 1)
        self.camera = None                  # with streaming: this camera's id in Kafka / the backend
        self.stream_ref = None              # ... and the current frame there: stream://<camera>/<session>/<frame>

    def push_frame(self, frame, t):
        self.buffer.append((t, frame))

    def handle(self, event, snapshot, t, save_clip=True):
        # cooldowns run on the detector's clock t (wall time for live cameras, video time for files), so a video
        # gives the same alerts however fast it is processed -- and the same as the streaming pipeline
        if t - self.last.get(event.key, -1e9) < self.cooldown:
            return False
        if t - self.last_type.get(event.kind, -1e9) < self.type_cooldown:
            return False                    # same kind of incident just reported (e.g. 3 people fighting = 1 alert)
        self.last[event.key] = t
        self.last_type[event.kind] = t
        attach = getattr(event, "attach", False)
        if attach and not self.stream_ref:
            return False                    # nothing to attach the evidence to
        now = time.time()
        if not attach:
            self.recent.append((now, event.message))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        day = self.dir / stamp[:8]
        day.mkdir(exist_ok=True)
        base = day / f"{stamp}_{event.key}"
        cv2.imwrite(f"{base}.jpg", snapshot)
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "type": event.kind,
            "message": event.message,
            "track_ids": event.track_ids,
            "snapshot": Path(f"{base}.jpg").relative_to(self.dir).as_posix(),     # "/" also on Windows (served by Linux)
        }
        if t < 1e8:                       # video file: t is the position in the video (live: t is the clock)
            record["video_time"] = round(t, 1)
        if self.camera:
            record["rover_id"] = self.camera
        if attach:                        # evidence for the stream's alert of this moment (the backend matches them)
            record["attach"] = True
            record["evidence"] = self.stream_ref
        if save_clip and len(self.buffer) > 1:
            frames = [(ft, f) for ft, f in self.buffer if ft >= t - self.clip_seconds]
            record["clip"] = Path(f"{base}.mp4").relative_to(self.dir).as_posix()
            threading.Thread(target=_write_clip, args=(f"{base}.mp4", frames), daemon=True).start()
        with self.log.open("a") as f:
            f.write(json.dumps(record) + "\n")

        at = f" (video {record['video_time']:.1f}s)" if "video_time" in record else ""
        if attach:
            print(f"[photo] {record['time']}{at}  photo + clip for the stream's alert: {event.message}")
        else:
            print(f"[ALERT] {record['time']}{at}  {event.message}")
            if self.sound:
                subprocess.Popen(["afplay", SOUND], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for listener in self.listeners:
            listener(record)
        return not attach

    def banner(self, seconds=6.0):
        now = time.time()
        return [m for ts, m in self.recent if now - ts < seconds]


def _write_clip(path, frames):
    if len(frames) < 2:
        return
    part = path[:-len(".mp4")] + ".part.mp4"          # appears under its real name only when it is complete
    span = frames[-1][0] - frames[0][0]
    fps = max(1.0, (len(frames) - 1) / span) if span > 0 else 10.0
    h, w = frames[0][1].shape[:2]
    # H.264 plays in browsers and on Android; fall back to MPEG-4 if this OpenCV build can't encode it
    out = cv2.VideoWriter(part, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not out.isOpened():
        out = cv2.VideoWriter(part, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _, f in frames:
        out.write(f)
    out.release()
    os.replace(part, path)
