from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str
    size: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "label": self.label, "size": self.size}
        if self.score is not None:
            data["score"] = self.score
        return data


COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Standard YOLO COCO datasets use contiguous class IDs 0-79. Torchvision uses
# the original sparse COCO category IDs, so it needs the gapped table above.
COCO80_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


SCENES = {
    "street": {
        "name": "Urban street",
        "boxes": [
            Box(58, 240, 138, 70, "car", "medium"),
            Box(232, 218, 48, 116, "person", "small"),
            Box(368, 230, 168, 82, "bus", "large"),
            Box(533, 104, 23, 56, "traffic light", "small"),
        ],
    },
    "warehouse": {
        "name": "Warehouse aisle",
        "boxes": [
            Box(80, 240, 130, 110, "box stack", "medium"),
            Box(305, 212, 178, 116, "forklift", "large"),
            Box(500, 275, 52, 42, "package", "small"),
            Box(170, 170, 46, 34, "label", "small"),
        ],
    },
}
