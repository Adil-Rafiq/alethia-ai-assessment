"""GPU detector wrapper and per-second aggregation."""

from collections import Counter

from . import CLASSES, CONF, IMGSZ, IOU, OBJECT_ORDER, PERSON_ID


class Detector:
    def __init__(self, weights, half=False, device=0):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.kw = dict(conf=CONF, iou=IOU, imgsz=IMGSZ, classes=list(CLASSES), half=half,
                       device=device, verbose=False)

    def __call__(self, imgs):
        """imgs: list of same-shape BGR arrays -> list of (set(labels), people_count)."""
        out = []
        for r in self.model.predict(imgs, **self.kw):
            cls = r.boxes.cls.int().tolist()
            out.append(({CLASSES[c] for c in cls}, cls.count(PERSON_ID)))
        return out


def aggregate(frames):
    """frames: list of (labels, people) for one second -> (objects, people) per brief section 5."""
    n = len(frames)
    if n == 0:
        return [], 0
    counts = Counter()
    for labels, _ in frames:
        counts.update(labels)
    objects = [o for o in OBJECT_ORDER if counts[o] * 2 >= n]
    p = sorted(pp for _, pp in frames)
    people = p[n // 2] if n % 2 else (p[n // 2 - 1] + p[n // 2]) // 2
    return objects, people
