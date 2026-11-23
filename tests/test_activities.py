"""Activity registry (activities.yaml) and plugins: the registry is consistent, every part of the pipeline reads it,
and a new activity can be added without touching the detector loop."""
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rover_ai import activities
from rover_ai.plugins import (
    AccessPlugin,
    ActivityPlugin,
    CrowdingPlugin,
    FightPlugin,
    FrameContext,
    build_plugins,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_registry_lists_every_activity_in_detection_order():
    acts = activities.load()
    assert list(acts) == ["loitering", "access", "fight", "theft", "crowding"]
    assert activities.alert_types(acts) == ("loitering", "restricted", "fight", "theft", "crowding")
    assert activities.models(acts) == {"fight": "fight-detector"}         # what the serving API loads
    fight = acts["fight"]
    assert fight.is_model and fight.stream and fight.model.train_stage == "train_fight"
    assert fight.retraining.dag == "fight_retraining" and fight.retraining.on_drift
    assert fight.retraining.stages == ("feedback", "features", "train_fight", "drift_reference")
    assert [a.name for a in activities.retrained(acts)] == ["fight"]


def test_the_repository_is_fully_wired():
    # plugin classes, config sections, DVC stages, Grafana rows, the registry name in params.yaml
    assert activities.check() == []


@pytest.mark.parametrize("change, message", [
    ({"kind": "learned"}, "kind must be one of"),
    ({"kind": "model"}, "needs model.registered_name"),
    ({"retraining": {"dag": "x"}}, "only model activities can be retrained"),
    ({"stream": True}, "only model activities are scored on the stream"),
    ({"plugin": "no_colon"}, "module:Class"),
    ({"alert_type": "fight"}, "share alert_type"),
])
def test_inconsistent_registries_are_rejected(tmp_path, change, message):
    data = yaml.safe_load((ROOT / "activities.yaml").read_text(encoding="utf-8"))
    data["activities"]["crowding"].update(change)
    path = tmp_path / "activities.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(activities.ActivityError, match=message):
        activities.load(path)


def test_plugins_are_built_for_enabled_activities_only():
    cfg = config()
    cfg["fight"]["method"] = "rules"                        # no model file needed
    names = [p.name for p in build_plugins(cfg, ROOT)]
    assert names == ["loitering", "access", "fight", "theft"]          # crowding is off by default
    cfg["crowding"]["enabled"] = True
    cfg["theft"]["enabled"] = False
    plugins = build_plugins(cfg, ROOT)
    assert [p.name for p in plugins] == ["loitering", "access", "fight", "crowding"]
    assert all(isinstance(p, ActivityPlugin) for p in plugins)
    assert plugins[-1].alert_type == "crowding"


def people(*ids):
    return [SimpleNamespace(id=i) for i in ids]


def test_crowding_alerts_once_per_gathering():
    plugin = CrowdingPlugin({"max_people": 2, "seconds": 3}, config(), ROOT)
    run = lambda t, n: plugin.update(FrameContext(tracks=people(*range(n)), t=t))        # noqa: E731
    assert run(0, 3) == [] and run(2.9, 3) == []                 # too short
    ctx = FrameContext(tracks=people(0, 1, 2), t=3.0)
    events = plugin.update(ctx)
    assert [e.kind for e in events] == ["crowding"] and events[0].track_ids == [0, 1, 2]
    assert ctx.flags[1] == {"CROWD"}
    assert run(10, 4) == []                                      # same gathering: no second alert
    assert run(11, 2) == []                                      # dispersed (not more than 2)
    assert run(12, 3) == [] and len(run(15, 3)) == 1             # a new gathering alerts again


def test_access_waits_for_a_parked_camera_and_fight_for_the_stream():
    cfg = config()
    zones = SimpleNamespace(has=lambda kind: True, contains=lambda *a: True)
    tr = SimpleNamespace(id=7, kpts=None, box=None, in_restricted=False)
    access = AccessPlugin(cfg["access"], cfg, ROOT)
    assert access.update(FrameContext(tracks=[tr], t=0, zones=zones, zones_active=False)) == []   # rover driving
    import rover_ai.rules as rules
    rules.anchor_point, saved = (lambda *a: (0, 0)), rules.anchor_point
    try:
        events = access.update(FrameContext(tracks=[tr], t=1, zones=zones))                          # parked
    finally:
        rules.anchor_point = saved
    assert [e.kind for e in events] == ["restricted"]

    fcfg = dict(cfg["fight"], method="rules")
    fight = FightPlugin(fcfg, cfg, ROOT)
    fight.name = "fight"
    fight.detector = SimpleNamespace(update=lambda *a: (["judged"], {7: 1.0}))
    ctx = FrameContext(tracks=[], t=0, server_scored=frozenset({"fight"}))
    assert fight.update(ctx) == [] and ctx.energy == {7: 1.0}         # no alert (the stream raises it), still drawn
    ctx = FrameContext(tracks=[], t=0)
    assert fight.update(ctx) == ["judged"] and ctx.energy == {7: 1.0}


@pytest.fixture()
def project(tmp_path):
    for f in ("activities.yaml", "config.yaml", "dvc.yaml", "params.yaml"):
        shutil.copy(ROOT / f, tmp_path / f)
    (tmp_path / "rover_ai").mkdir()
    dash = tmp_path / "docker" / "grafana" / "dashboards"
    dash.mkdir(parents=True)
    shutil.copy(ROOT / "docker/grafana/dashboards/rover-model.json", dash / "rover-model.json")
    return tmp_path


def test_scaffold_adds_a_disabled_activity_that_plugs_in(project):
    files = activities.scaffold("running", "rule", "Running", "someone runs through the area", root=project)
    assert [f.name for f in files] == ["activities.yaml", "config.yaml", "running.py"]
    acts = activities.load(project / "activities.yaml")
    run = acts["running"]
    assert run.kind == "rule" and run.alert_type == "running" and run.plugin == "rover_ai.custom.running:RunningPlugin"
    cfg = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["running"]["enabled"] is False

    spec = importlib.util.spec_from_file_location("running_plugin", project / "rover_ai/custom/running.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = module.RunningPlugin(cfg["running"], cfg, project)
    assert isinstance(plugin, ActivityPlugin)
    assert plugin.update(FrameContext(tracks=people(1, 2), t=0)) == []      # condition is a TODO: never fires
    module.RunningPlugin.condition = lambda self, tr, ctx: tr.id == 2
    plugin = module.RunningPlugin(dict(cfg["running"], seconds=1), cfg, project)
    assert plugin.update(FrameContext(tracks=people(1, 2), t=0)) == []
    assert [e.message for e in plugin.update(FrameContext(tracks=people(1, 2), t=1))] == ["RUNNING: ID 2"]

    problems = activities.check(root=project, import_plugins=False)
    assert problems == ["running: no Grafana row 'Activity: Running' (run python docker/grafana/build_dashboards.py)"]
    with pytest.raises(activities.ActivityError, match="already"):
        activities.scaffold("running", "rule", "Running", root=project)


def test_scaffolded_model_activity_names_its_missing_training_stage(project):
    activities.scaffold("vandalism", "model", "Vandalism", root=project)
    vandalism = activities.load(project / "activities.yaml")["vandalism"]
    assert vandalism.model.registered_name == "vandalism-detector" and vandalism.retraining is None
    problems = activities.check(root=project, import_plugins=False)
    assert "vandalism: model.train_stage 'train_vandalism' is not a stage in dvc.yaml" in problems
