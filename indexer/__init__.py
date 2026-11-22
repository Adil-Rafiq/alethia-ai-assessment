"""Video indexing package: decode workers, detector, and per-second aggregation."""

# COCO class id -> output name (YOLO11 is trained on COCO-80).
CLASSES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
OBJECT_ORDER = ["person", "car", "dog", "laptop", "cell phone"]
PERSON_ID = 0
IMGSZ = 640
CONF = 0.4
IOU = 0.7
