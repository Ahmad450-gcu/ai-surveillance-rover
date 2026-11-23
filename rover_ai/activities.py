"""Activity registry (activities.yaml): the one place that says which suspicious activities exist and how each one is
detected, served, streamed, retrained and monitored. Every part of the pipeline reads it from here.

    python -m rover_ai.activities list                      what is registered, and where each activity runs
    python -m rover_ai.activities check                     is every activity fully wired? (CI runs this)
    python -m rover_ai.activities new loiter_run --kind rule --title "Running"     scaffold a new activity

This module only needs PyYAML, so the small service images (serving, backend) can read the registry too.
"""
import argparse
import importlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "activities.yaml"
KINDS = ("rule", "model")
INPUTS = ("person_pair", "person", "scene")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
STREAM_SCORERS = {"fight"}      # activities the Spark job has a stateful scorer for (streaming/scoring.py)


class ActivityError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    registered_name: str
    input: str = "person_pair"
    window_frames: int = 20
    train_stage: str | None = None
    metrics: str | None = None


@dataclass(frozen=True)
class RetrainSpec:
    dag: str
    schedule: str | None = "0 3 * * *"
    on_drift: bool = True
    min_new_labels: int = 5
    max_age_days: float = 7
    prepare: str | None = None
    stages: tuple = ()


@dataclass(frozen=True)
class Activity:
    name: str
    title: str
    kind: str
    plugin: str
    config: str
    alert_type: str
    description: str = ""
    model: ModelSpec | None = None
    stream: bool = False
    retraining: RetrainSpec | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_model(self):
        return self.kind == "model"

    def plugin_class(self):
        module, _, cls = self.plugin.partition(":")
        return getattr(importlib.import_module(module), cls)


def _build(name, raw):
    if not NAME_RE.match(name):
        raise ActivityError(f"{name}: activity names are lower case letters, digits and _ (2-31 characters)")
    if not isinstance(raw, dict):
        raise ActivityError(f"{name}: expected a mapping")
    raw = dict(raw)
    missing = [k for k in ("title", "kind", "plugin", "config", "alert_type") if not raw.get(k)]
    if missing:
        raise ActivityError(f"{name}: missing {', '.join(missing)}")
    if raw["kind"] not in KINDS:
        raise ActivityError(f"{name}: kind must be one of {KINDS}, not {raw['kind']!r}")
    if ":" not in raw["plugin"]:
        raise ActivityError(f"{name}: plugin must be 'module:Class', not {raw['plugin']!r}")
    model = raw.pop("model", None)
    if raw["kind"] == "model":
        if not isinstance(model, dict) or not model.get("registered_name"):
            raise ActivityError(f"{name}: a model activity needs model.registered_name")
        model = ModelSpec(**model)
        if model.input not in INPUTS:
            raise ActivityError(f"{name}: model.input must be one of {INPUTS}")
    elif model:
        raise ActivityError(f"{name}: only kind 'model' has a model section")
    retraining = raw.pop("retraining", None)
    if retraining is not None:
        if raw["kind"] != "model":
            raise ActivityError(f"{name}: only model activities can be retrained")
        retraining = dict(retraining)
        retraining.setdefault("dag", f"{name}_retraining")
        retraining["stages"] = tuple(retraining.get("stages") or ())
        retraining = RetrainSpec(**retraining)
    if raw.get("stream") and raw["kind"] != "model":
        raise ActivityError(f"{name}: only model activities are scored on the stream")
    known = {f for f in Activity.__dataclass_fields__} - {"name", "model", "retraining", "extra"}
    extra = {k: raw.pop(k) for k in list(raw) if k not in known}
    return Activity(name=name, model=model, retraining=retraining, extra=extra, **raw)


def load(path=None):
    """activities.yaml -> {name: Activity} in file order. Raises ActivityError if the registry is inconsistent."""
    path = Path(path or REGISTRY)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("activities")
    if not isinstance(raw, dict) or not raw:
        raise ActivityError(f"{path}: no 'activities' mapping")
    acts = {name: _build(name, spec) for name, spec in raw.items()}
    for attr in ("alert_type", "config"):
        seen = {}
        for a in acts.values():
            if getattr(a, attr) in seen:
                raise ActivityError(f"{a.name} and {seen[getattr(a, attr)]} share {attr} {getattr(a, attr)!r}")
            seen[getattr(a, attr)] = a.name
    dags = [a.retraining.dag for a in acts.values() if a.retraining]
    if len(dags) != len(set(dags)):
        raise ActivityError(f"two activities use the same retraining DAG id: {dags}")
    return acts


def alert_types(acts=None):
    return tuple(a.alert_type for a in (acts or load()).values())


def models(acts=None):
    """{activity: registered model name} of every model activity (what the serving API loads)."""
    return {a.name: a.model.registered_name for a in (acts or load()).values() if a.is_model}


def retrained(acts=None):
    return [a for a in (acts or load()).values() if a.retraining]


def by_alert_type(acts=None):
    return {a.alert_type: a for a in (acts or load()).values()}


# ----------------------------------------------------------------------------------------------------------- check
def check(acts=None, root=ROOT, import_plugins=True):
    """Every way an activity can be half-wired -> list of problems (empty = everything is consistent)."""
    problems = []
    try:
        acts = acts or load(root / "activities.yaml")
    except ActivityError as e:
        return [str(e)]
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    dvc = yaml.safe_load((root / "dvc.yaml").read_text(encoding="utf-8")).get("stages", {})
    params = yaml.safe_load((root / "params.yaml").read_text(encoding="utf-8"))
    dash = root / "docker" / "grafana" / "dashboards" / "rover-model.json"
    titles = {p.get("title") for p in json.loads(dash.read_text(encoding="utf-8")).get("panels", [])} if dash.exists() else set()
    for a in acts.values():
        section = config.get(a.config)
        if not isinstance(section, dict) or "enabled" not in section:
            problems.append(f"{a.name}: config.yaml has no section '{a.config}' with an 'enabled' switch")
        if import_plugins:
            try:
                from rover_ai.plugins import ActivityPlugin
                cls = a.plugin_class()
                if not (isinstance(cls, type) and issubclass(cls, ActivityPlugin)):
                    problems.append(f"{a.name}: {a.plugin} is not an ActivityPlugin")
            except Exception as e:                                         # noqa: BLE001 - report, don't crash
                problems.append(f"{a.name}: plugin {a.plugin} can't be imported ({type(e).__name__}: {e})")
        if f"Activity: {a.title}" not in titles:
            problems.append(f"{a.name}: no Grafana row 'Activity: {a.title}' (run python docker/grafana/build_dashboards.py)")
        if a.is_model:
            stage = a.model.train_stage
            if not stage or stage not in dvc:
                problems.append(f"{a.name}: model.train_stage {stage!r} is not a stage in dvc.yaml")
            if a.name == "fight" and params.get("registry", {}).get("model_name") != a.model.registered_name:
                problems.append(f"{a.name}: params.yaml registry.model_name differs from {a.model.registered_name}")
        if a.stream and a.name not in STREAM_SCORERS:
            problems.append(f"{a.name}: stream: true, but the Spark job has no scorer for it (streaming/scoring.py)")
        if a.retraining:
            for s in a.retraining.stages:
                if s not in dvc:
                    problems.append(f"{a.name}: retraining stage {s!r} is not in dvc.yaml")
            if a.model.train_stage and a.model.train_stage not in a.retraining.stages:
                problems.append(f"{a.name}: retraining.stages must include the training stage {a.model.train_stage}")
    return problems


# --------------------------------------------------------------------------------------------------------- scaffold
PLUGIN_TEMPLATE = '''"""{title} ({kind} activity, created by python -m rover_ai.activities new). See docs/ACTIVITIES.md."""
from rover_ai.events import Event
from rover_ai.plugins import ActivityPlugin


class {cls}(ActivityPlugin):
    """{description}"""

    def __init__(self, cfg, full_cfg, root):
        super().__init__(cfg, full_cfg, root)
        self.seconds = float(cfg.get("seconds", 5))      # every key of config.yaml section "{config}" is in cfg
        self.since = {{}}                                 # track id -> when the condition started
        self.alerted = set()
{model_init}
    def update(self, ctx):
        """Called on every frame. ctx.tracks: people (id, box, 17 keypoints, history); ctx.items: objects;
        ctx.t: time in seconds; ctx.flags[track_id]: labels drawn next to a person. Return a list of Event."""
        events = []
        for tr in ctx.tracks:
            if not self.condition(tr, ctx):
                self.since.pop(tr.id, None)
                self.alerted.discard(tr.id)
                continue
            start = self.since.setdefault(tr.id, ctx.t)
            if ctx.t - start >= self.seconds:
                ctx.flags[tr.id].add("{flag}")
                if tr.id not in self.alerted:
                    self.alerted.add(tr.id)
                    events.append(Event("{alert_type}", f"{name}-{{tr.id}}", f"{flag}: ID {{tr.id}}", [tr.id]))
        return events

    def condition(self, track, ctx):
        """TODO: when is this person doing "{title}"?"""
{condition}
'''

RULE_CONDITION = "        return False\n"
MODEL_INIT = '''        # Model activity: the serving API answers with the registry's @champion for this activity.
        # from rover_ai.learned import RemoteModel
        # self.model = RemoteModel(cfg.get("model", "http://127.0.0.1:8000/v1/models/{name}"))
'''
MODEL_CONDITION = ("        # build the model input from track.hist (see rover_ai/learned.py window_features), then:\n"
                   "        # return self.model(x)[0] >= cfg threshold\n        return False\n")


def _insert_entry(text, name, entry):
    if re.search(rf"^  {re.escape(name)}:\s*$", text, re.M):
        raise ActivityError(f"{name} is already registered")
    return text.rstrip("\n") + "\n\n" + entry


def scaffold(name, kind, title, description="", alert_type=None, root=ROOT):
    """Create a disabled activity: registry entry, config section, plugin module (and a model block). -> files"""
    if not NAME_RE.match(name):
        raise ActivityError("activity names are lower case letters, digits and _")
    if kind not in KINDS:
        raise ActivityError(f"kind must be one of {KINDS}")
    alert_type = alert_type or name
    acts = load(root / "activities.yaml")
    if name in acts or alert_type in alert_types(acts):
        raise ActivityError(f"{name} / alert type {alert_type} already exists")
    cls = "".join(p.capitalize() for p in name.split("_")) + "Plugin"
    module = f"rover_ai.custom.{name}"
    description = description or f"TODO: describe when '{title}' is raised"
    q = lambda text: json.dumps(text, ensure_ascii=False)        # noqa: E731 - a YAML-safe quoted string
    entry = (f"  {name}:\n    title: {q(title)}\n    kind: {kind}\n    description: {q(description)}\n"
             f"    plugin: {module}:{cls}\n    config: {name}\n    alert_type: {alert_type}\n")
    if kind == "model":
        entry += (f"    model:\n      registered_name: {name.replace('_', '-')}-detector\n      input: person\n"
                  f"      window_frames: 20\n      train_stage: train_{name}   # TODO: add this stage to dvc.yaml\n"
                  f"    # retraining:                     # uncomment once the training stage exists\n"
                  f"    #   dag: {name}_retraining\n    #   stages: [features, train_{name}]\n")
    reg = root / "activities.yaml"
    reg.write_text(_insert_entry(reg.read_text(encoding="utf-8"), name, entry), encoding="utf-8")

    cfg = root / "config.yaml"
    text = cfg.read_text(encoding="utf-8")
    if re.search(rf"^{re.escape(name)}:", text, re.M):
        raise ActivityError(f"config.yaml already has a section {name}")
    cfg.write_text(text.rstrip("\n") + f"\n\n{name}:                    # activity '{title}' (activities.yaml)\n"
                   f"  enabled: false                    # switch on when the plugin is ready\n  seconds: 5\n",
                   encoding="utf-8")

    pkg = root / "rover_ai" / "custom"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").touch()
    plugin = pkg / f"{name}.py"
    plugin.write_text(PLUGIN_TEMPLATE.format(
        title=title, kind=kind, cls=cls, description=description, config=name, alert_type=alert_type, name=name,
        flag=name.upper().replace("_", " "), model_init=MODEL_INIT.format(name=name) if kind == "model" else "",
        condition=MODEL_CONDITION if kind == "model" else RULE_CONDITION), encoding="utf-8")
    return [reg, cfg, plugin]


# --------------------------------------------------------------------------------------------------------------- CLI
def _where(a):
    parts = ["edge"]
    if a.is_model:
        parts.append(f"serving /v1/models/{a.name}")
    if a.stream:
        parts.append("stream")
    if a.retraining:
        parts.append(f"Airflow {a.retraining.dag}")
    return ", ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("check")
    p = sub.add_parser("new", help="scaffold a new (disabled) activity")
    p.add_argument("name")
    p.add_argument("--kind", choices=KINDS, default="rule")
    p.add_argument("--title")
    p.add_argument("--description", default="")
    p.add_argument("--alert-type")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        for a in load().values():
            on = "on " if (config.get(a.config) or {}).get("enabled") else "off"
            model = f" model {a.model.registered_name}" if a.is_model else ""
            print(f"{a.name:<12} {on} {a.kind:<5} alert '{a.alert_type}'{model}  ->  {_where(a)}")
        return 0
    if args.cmd == "check":
        problems = check()
        for p in problems:
            print("PROBLEM:", p)
        print(f"{len(load()) if not problems else '?'} activities, {len(problems)} problem(s)")
        return 1 if problems else 0
    files = scaffold(args.name, args.kind, args.title or args.name.replace("_", " ").capitalize(),
                     args.description, args.alert_type)
    print("created / changed:", *[f"  {f.relative_to(ROOT)}" for f in files], sep="\n")
    print(f"next: write the condition in {files[-1].relative_to(ROOT)}, run python docker/grafana/build_dashboards.py,"
          f" then python -m rover_ai.activities check; switch it on with --set {args.name}.enabled=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
