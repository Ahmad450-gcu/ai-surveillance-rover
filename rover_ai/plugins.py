"""Activity plugins: one common interface for every suspicious activity, so the detector loop (run.py) doesn't know
or care which activities exist. activities.yaml says which plugin class implements each activity.

    class MyPlugin(ActivityPlugin):
        def update(self, ctx):          # every frame
            return [Event(...), ...]

A plugin gets its own config.yaml section (cfg), the whole config (full_cfg) and the project root. The FrameContext
carries everything a detector may look at: people tracks, objects, zones, time, and the per-person flags it can set
(drawn next to the person on screen).
"""
from collections import defaultdict
from dataclasses import dataclass, field

from .events import Event


@dataclass
class FrameContext:
    tracks: list                                     # visible people (rover_ai.tracks.Track): id, box, kpts, history
    t: float                                         # seconds (video time for files, wall time for live cameras)
    shape: tuple = (480, 640, 3)                     # frame shape
    zones: object = None                             # rover_ai.zones.Zones
    zones_active: bool = True                        # zones only mean something while the camera is parked
    items: list = field(default_factory=list)        # detected objects (theft)
    fresh: bool = False                              # objects were detected on this frame (every Nth frame)
    flags: dict = field(default_factory=lambda: defaultdict(set))
    energy: dict = field(default_factory=dict)       # debug values per person (fight motion)
    server_scored: frozenset = frozenset()           # activities the stream scores right now (skip them here)


class ActivityPlugin:
    """Base class. name / alert_type are set from activities.yaml by build_plugins()."""
    name = "activity"
    alert_type = "activity"

    def __init__(self, cfg, full_cfg, root):
        self.cfg, self.full_cfg, self.root = cfg, full_cfg, root

    def update(self, ctx):
        """-> list of Event for this frame."""
        raise NotImplementedError

    def static(self, persons, zones, shape, flags):
        """Single image (no motion, no tracking): -> list of Event. Default: nothing can be judged."""
        return []

    def reset(self):
        """The camera moved (rover driving): forget anything tied to positions in the picture."""

    def item_status(self, item):
        """Text drawn next to a detected object."""
        return ""

    def describe(self):
        return self.name


def build_plugins(cfg, root, acts=None, only_enabled=True):
    """Plugins of every registered activity whose config.yaml section is enabled, in registry order."""
    from . import activities
    acts = acts or activities.load()
    plugins = []
    for a in acts.values():
        section = cfg.get(a.config) or {}
        if only_enabled and not section.get("enabled", False):
            continue
        plugin = a.plugin_class()(section, cfg, root)
        plugin.name, plugin.alert_type = a.name, a.alert_type
        plugins.append(plugin)
    return plugins


# ------------------------------------------------------------------------------------------------ built-in plugins
class LoiterPlugin(ActivityPlugin):
    def __init__(self, cfg, full_cfg, root):
        super().__init__(cfg, full_cfg, root)
        from .rules import LoiterRule
        self.rule = LoiterRule(cfg, full_cfg["zones"]["anchor"])

    def update(self, ctx):
        return self.rule.update(ctx.tracks, ctx.zones, ctx.t, ctx.shape, ctx.flags)


class AccessPlugin(ActivityPlugin):
    def __init__(self, cfg, full_cfg, root):
        super().__init__(cfg, full_cfg, root)
        from .rules import AccessRule
        self.anchor = full_cfg["zones"]["anchor"]
        self.rule = AccessRule(self.anchor)

    def update(self, ctx):
        return self.rule.update(ctx.tracks, ctx.zones, ctx.t, ctx.shape, ctx.flags) if ctx.zones_active else []

    def static(self, persons, zones, shape, flags):
        from .rules import static_access
        return static_access(persons, zones, shape, self.anchor, flags)


def make_fight_detector(fcfg, root):
    """Trained model if available (fight.method: model), otherwise the hand-tuned rules."""
    from .fight import FightDetector
    if fcfg.get("method", "rules") == "model":
        from .learned import LearnedFightDetector, is_remote
        ref = fcfg.get("model", "models/fight_model.onnx")
        path = root / ref if not is_remote(ref) else None
        if path is None or path.exists():
            try:
                det = LearnedFightDetector(fcfg, root)
            except OSError as e:                                  # serving API not reachable at start-up
                print(f"Fight detection: serving API {ref} unreachable ({e}), using hand-tuned rules")
                return FightDetector(fcfg)
            reg = det.meta.get("registry", {})
            ver = f"{reg['model']} v{reg['version']} ({reg['alias']}), " if reg else ""
            print(f"Fight detection: trained {det.meta['arch'].get('type', 'gru').upper()} "
                  f"{ref if path is None else path.name} ({ver}threshold {det.thr:.2f})")
            return det
        print(f"Fight detection: {path.name} not found, using hand-tuned rules "
              "(deploy the registry champion: python -m mlops.registry pull)")
    return FightDetector(fcfg)


class FightPlugin(ActivityPlugin):
    def __init__(self, cfg, full_cfg, root):
        super().__init__(cfg, full_cfg, root)
        self.detector = make_fight_detector(cfg, root)

    def update(self, ctx):
        # with streaming, the stream judges fights; this computer does it only while Kafka is unreachable
        if self.name in ctx.server_scored:
            return []
        events, energy = self.detector.update(ctx.tracks, ctx.t, ctx.flags)
        ctx.energy.update(energy)
        return events

    def static(self, persons, zones, shape, flags):
        from .fight import static_fight
        return static_fight(persons, self.cfg["proximity"], flags)


class TheftPlugin(ActivityPlugin):
    def __init__(self, cfg, full_cfg, root):
        super().__init__(cfg, full_cfg, root)
        from .theft import TheftDetector
        self.detector = TheftDetector(cfg)

    def update(self, ctx):
        return self.detector.update(ctx.items, ctx.tracks, ctx.t, ctx.fresh, ctx.flags)

    def reset(self):
        self.detector.reset()

    def item_status(self, item):
        return self.detector.status(item)


class CrowdingPlugin(ActivityPlugin):
    """More than max_people people in view for `seconds`: one alert per gathering (it must disperse below the limit
    before the next one). An example of an activity added through the registry: this class + a config section +
    one entry in activities.yaml; nothing else in the pipeline changed."""

    def __init__(self, cfg, full_cfg, root):
        super().__init__(cfg, full_cfg, root)
        self.max_people = int(cfg.get("max_people", 5))
        self.seconds = float(cfg.get("seconds", 10))
        self.since, self.alerted = None, False

    def update(self, ctx):
        n = len(ctx.tracks)
        if n <= self.max_people:
            self.since, self.alerted = None, False
            return []
        if self.since is None:
            self.since = ctx.t
        if ctx.t - self.since < self.seconds:
            return []
        for tr in ctx.tracks:
            ctx.flags[tr.id].add("CROWD")
        if self.alerted:
            return []
        self.alerted = True
        return [Event("crowding", "crowding", f"CROWDING: {n} people in view for {ctx.t - self.since:.0f}s",
                      [tr.id for tr in ctx.tracks])]

    def static(self, persons, zones, shape, flags):
        if len(persons) <= self.max_people:
            return []
        return [Event("crowding", "crowding-img", f"CROWDING: {len(persons)} people in the picture")]
