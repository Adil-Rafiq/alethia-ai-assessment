"""One-time preparation (untimed, has internet): download yolo11l.pt and build a TensorRT FP16 engine.

python prepare.py
The engine is GPU-specific, so it must be built on the machine that runs run.py.
"""

import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    # TensorRT 11 export runs ModelOpt AutoCast, which runs an FP32 ONNX Runtime pass on CPU at the max
    # batch; batch 32 and 16 were OOM-killed with 12 GB RAM, 8 works. Must be >= run.py --batch.
    ap.add_argument("--batch", type=int, default=8, help="max batch size of the engine")
    ap.add_argument("--no-engine", action="store_true", help="only download weights")
    a = ap.parse_args()

    from ultralytics import YOLO

    pt = os.path.join(HERE, "yolo11l.pt")
    if not os.path.exists(pt):
        YOLO("yolo11l.pt")  # downloads into cwd
        if os.path.exists("yolo11l.pt") and os.path.abspath("yolo11l.pt") != pt:
            shutil.move("yolo11l.pt", pt)
    print("weights:", pt)
    if a.no_engine:
        return

    # dynamic=True keeps Ultralytics' minimal-rectangle letterbox (e.g. 640x384), the same padding the
    # .pt model uses, so results match the every-frame PyTorch reference.
    out = YOLO(pt).export(format="engine", imgsz=640, quantize=16, dynamic=True, batch=a.batch,
                          simplify=True, device=0)
    print("engine:", out)


if __name__ == "__main__":
    main()
