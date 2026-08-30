from __future__ import annotations

import base64
import io
import numpy as np

from .config import HEIGHT, WIDTH
from .domain import Box

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    raise SystemExit("Pillow is required. Install the project dependencies in the selected environment.") from exc


def make_scene(scene_id: str) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#b9d6df")
    draw = ImageDraw.Draw(image)
    if scene_id == "warehouse":
        draw.rectangle((0, 0, WIDTH, HEIGHT), fill="#d9ded5")
        draw.rectangle((0, 332, WIDTH, HEIGHT), fill="#777f78")
        for x in range(0, WIDTH, 80):
            draw.rectangle((x, 0, x + 18, 210), fill="#b6c0b7")
        for x, y, w, h, color in [(80, 286, 62, 64, "#b97945"), (142, 275, 68, 75, "#d19753"), (98, 240, 68, 50, "#c88c52")]:
            draw.rectangle((x, y, x + w, y + h), fill=color, outline="#7a5732", width=2)
        draw.rounded_rectangle((305, 252, 408, 301), radius=8, fill="#d6a63b")
        draw.rectangle((375, 212, 423, 282), outline="#2d3230", width=8)
        draw.line((433, 222, 433, 328, 483, 328), fill="#2d3230", width=8)
        draw.ellipse((321, 295, 353, 327), fill="#222")
        draw.ellipse((385, 292, 421, 328), fill="#222")
        draw.rectangle((500, 275, 552, 317), fill="#bd7b42", outline="#7a5732", width=2)
        draw.rectangle((170, 170, 216, 204), fill="#f0df96", outline="#7a5732", width=2)
    else:
        draw.rectangle((0, 0, WIDTH, 198), fill="#b9d6df")
        draw.rectangle((0, 198, WIDTH, 210), fill="#cbd3c8")
        draw.rectangle((0, 210, WIDTH, HEIGHT), fill="#4b504d")
        draw.rectangle((0, 332, WIDTH, 340), fill="#ece8d8")
        draw.rounded_rectangle((58, 260, 196, 295), radius=10, fill="#d05d3f")
        draw.rounded_rectangle((86, 240, 138, 267), radius=8, fill="#b8d3dc")
        draw.ellipse((84, 287, 104, 307), fill="#222")
        draw.ellipse((165, 287, 185, 307), fill="#222")
        draw.ellipse((244, 218, 268, 242), fill="#51392f")
        draw.rounded_rectangle((242, 244, 270, 293), radius=8, fill="#2f6c83")
        draw.line((256, 290, 241, 334), fill="#252525", width=6)
        draw.line((256, 290, 271, 334), fill="#252525", width=6)
        draw.rounded_rectangle((368, 230, 536, 294), radius=8, fill="#e0b84d")
        for i in range(4):
            draw.rectangle((386 + i * 34, 244, 410 + i * 34, 266), fill="#49616b")
        draw.ellipse((391, 282, 415, 306), fill="#222")
        draw.ellipse((489, 282, 513, 306), fill="#222")
        draw.rounded_rectangle((533, 104, 556, 160), radius=5, fill="#283232")
        for i, color in enumerate(["#d94735", "#e2b736", "#3aa765"]):
            draw.ellipse((539, 111 + i * 16, 550, 122 + i * 16), fill=color)
    return image


DEGRADATION_LEVELS = {
    "blur": {
        1: {"kernelLength": 3, "angleDegrees": 12},
        2: {"kernelLength": 7, "angleDegrees": 12},
        3: {"kernelLength": 11, "angleDegrees": 12},
        4: {"kernelLength": 15, "angleDegrees": 12},
        5: {"kernelLength": 21, "angleDegrees": 12},
    },
    "lowlight": {
        1: {"exposure": 0.78, "gamma": 1.20, "noiseSigma": 2.0},
        2: {"exposure": 0.65, "gamma": 1.35, "noiseSigma": 4.0},
        3: {"exposure": 0.52, "gamma": 1.50, "noiseSigma": 6.0},
        4: {"exposure": 0.40, "gamma": 1.70, "noiseSigma": 8.0},
        5: {"exposure": 0.30, "gamma": 1.90, "noiseSigma": 10.0},
    },
    "jpeg": {
        1: {"quality": 85},
        2: {"quality": 70},
        3: {"quality": 50},
        4: {"quality": 30},
        5: {"quality": 15},
    },
}


def degradation_parameters(kind: str, severity: int) -> dict[str, float | int | str]:
    severity = max(0, min(5, int(severity)))
    if severity == 0:
        return {"kind": kind, "severity": 0, "identity": True}
    return {"kind": kind, "severity": severity, **DEGRADATION_LEVELS.get(kind, {}).get(severity, {})}


def _motion_blur(image: Image.Image, length: int, angle_degrees: float) -> Image.Image:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for motion blur") from exc
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    centre = ((length - 1) / 2, (length - 1) / 2)
    rotation = cv2.getRotationMatrix2D(centre, angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel_sum = float(kernel.sum())
    kernel = kernel / kernel_sum if kernel_sum else kernel
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    blurred = cv2.filter2D(array, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return Image.fromarray(blurred, mode="RGB")


def _lowlight(image: Image.Image, exposure: float, gamma: float, noise_sigma: float, seed: int) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    darkened = np.power(np.clip(array * exposure, 0.0, 1.0), gamma)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma / 255.0, size=darkened.shape)
    result = np.clip((darkened + noise) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def degrade(image: Image.Image, kind: str, severity: int, seed: int = 0) -> Image.Image:
    severity = max(0, min(5, severity))
    if severity == 0:
        return image.copy()
    parameters = DEGRADATION_LEVELS.get(kind, {}).get(severity)
    if parameters is None:
        return image.copy()
    if kind == "blur":
        return _motion_blur(image, int(parameters["kernelLength"]), float(parameters["angleDegrees"]))
    if kind == "lowlight":
        return _lowlight(
            image,
            float(parameters["exposure"]),
            float(parameters["gamma"]),
            float(parameters["noiseSigma"]),
            seed,
        )
    if kind == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(parameters["quality"]), optimize=False)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return image.copy()


def draw_boxes(image: Image.Image, boxes: list[Box], color: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    visible_boxes = [box for box in boxes if box.score is None or box.score >= 0.35][:20]
    for box in visible_boxes:
        xy = (box.x, box.y, box.x + box.w, box.y + box.h)
        draw.rectangle(xy, outline=color, width=3)
        label = f"{box.label} {box.score:.2f}" if box.score is not None else box.label
        draw.rectangle((box.x, max(0, box.y - 18), box.x + len(label) * 7 + 8, box.y), fill=color)
        draw.text((box.x + 4, max(0, box.y - 16)), label, fill="white")
    return out


def image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_from_data_url(value: str) -> Image.Image:
    if "," in value:
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image.thumbnail((960, 720))
    return image


def bytes_from_data_url(value: str) -> bytes:
    if "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def box_size_from_area(area: float) -> str:
    return "small" if area < 32 * 32 else "medium" if area < 96 * 96 else "large"


def stable_seed(seed: str) -> int:
    h = 2166136261
    for char in seed:
        h ^= ord(char)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def stable_random(seed: str) -> float:
    return (stable_seed(seed) % 10000) / 10000
