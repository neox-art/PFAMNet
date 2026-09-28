from dataclasses import dataclass, field


@dataclass
class PromptPoint:
    x: float
    y: float
    label: int


@dataclass
class PromptBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class PromptState:
    positive_points: list[PromptPoint] = field(default_factory=list)
    negative_points: list[PromptPoint] = field(default_factory=list)
    boxes: list[PromptBox] = field(default_factory=list)

    def clear(self):
        self.positive_points.clear()
        self.negative_points.clear()
        self.boxes.clear()

    def to_payload(self):
        return {
            "positive_points": [p.__dict__ for p in self.positive_points],
            "negative_points": [p.__dict__ for p in self.negative_points],
            "boxes": [b.__dict__ for b in self.boxes],
        }

    def has_prompt(self):
        return bool(self.positive_points or self.negative_points or self.boxes)
