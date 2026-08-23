from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Segment:
    id: int
    start: float
    end: float
    japanese: str
    chinese: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        return cls(int(value["id"]), float(value["start"]), float(value["end"]),
                   str(value.get("japanese", value.get("text", ""))),
                   str(value.get("chinese", "")))
