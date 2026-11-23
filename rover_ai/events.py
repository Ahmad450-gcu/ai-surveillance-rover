from dataclasses import dataclass, field


@dataclass
class Event:
    kind: str                       # alert type of the activity (activities.yaml): loitering, restricted, fight, ...
    key: str                        # used for de-duplication / cooldown
    message: str
    track_ids: list = field(default_factory=list)
