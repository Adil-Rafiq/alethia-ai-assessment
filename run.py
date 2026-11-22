"""Index FineVideo parquet files: per-second presence of 5 objects + people count (YOLO11l).

python run.py --input <folder of .parquet files> --output results.jsonl
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import defaultdict

import pyarrow.parquet as pq

from indexer.decode import worker_main

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Writer:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.lock = threading.Lock()

    def lines(self, objs):
        with self.lock:
            self.f.write("".join(json.dumps(o) + "\n" for o in objs))
            self.f.flush()

    def close(self):
        self.f.close()


def reader(files, task_q, writer, n_workers, max_videos, stats):
    """Stream mp4 bytes row by row (never holds a whole 0.5 GB file)."""
    count = 0
    for path in files:
        name, row = os.path.basename(path), 0
        try:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                for val in batch.column(0):
                    if max_videos and count >= max_videos:
                        break
                    vid = f"{name}:{row}"
                    row += 1
                    data = val.as_py()
                    if isinstance(data, dict):
                        data = data.get("bytes")
                    if not data:
                        writer.lines([{"video": vid, "error": "empty mp4 field"}])
                        continue
                    task_q.put((vid, data))
                    count += 1
                    stats["queued"] = count
        except Exception as e:
            writer.lines([{"video": f"{name}:{row}", "error": f"parquet read failed: {type(e).__name__}"}])
        if max_videos and count >= max_videos:
            break
    for _ in range(n_workers):
        task_q.put(None)


def default_weights():
    for w in ("yolo11l.engine", "yolo11l.pt"):
        p = os.path.join(HERE, w)
        if os.path.exists(p):
            return p
    return "yolo11l.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=3, help="frames sampled per second (0 = every frame)")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--batch", type=int, default=8, help="must be <= the engine's max batch (prepare.py)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--fp32", action="store_true", help="FP32 instead of FP16 (reference runs)")
    ap.add_argument("--skip-nonref", action="store_true", help="don't decode non-reference frames")
    ap.add_argument("--max-videos", type=int, default=0, help="dev: stop after N videos")
    ap.add_argument("--max-seconds", type=int, default=0, help="dev: only first N seconds per video")
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    log(f"{len(files)} parquet files, k={args.k}, workers={args.workers}, batch={args.batch}, "
        f"fp32={args.fp32}, skip_nonref={args.skip_nonref}")
    writer = Writer(args.output)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=args.workers * 2)
    out_q = ctx.Queue(maxsize=256)

    def spawn(wid):
        p = ctx.Process(target=worker_main, args=(wid, task_q, out_q, args.k, args.max_seconds, args.skip_nonref),
                        daemon=True)
        p.start()
        return p

    procs = {wid: spawn(wid) for wid in range(args.workers)}  # spawn before CUDA init
    stats = {"queued": 0}
    threading.Thread(target=reader, args=(files, task_q, writer, args.workers, args.max_videos, stats),
                     daemon=True).start()

    from indexer.detect import Detector, aggregate  # imports torch (after workers started)

    det = Detector(args.weights or default_weights(), half=not args.fp32)
    tm = {"gpu": 0.0, "wait": 0.0}  # where the main process spends its time
    t_ready = time.time()
    log(f"model loaded in {t_ready - t0:.1f}s")

    videos = {}  # vid -> {"secs": {sec: [(labels, people)]}, "inflight": int, "end": (last, err)|None}
    current = {}  # wid -> vid being decoded
    buckets = defaultdict(list)  # frame shape -> [(vid, sec, img)]
    wdone, n_videos, n_secs, n_frames = 0, 0, 0, 0
    last_check = time.time()

    def maybe_finish(vid):
        nonlocal n_videos, n_secs
        v = videos.get(vid)
        if v is None or v["end"] is None or v["inflight"] > 0:
            return
        last, err = v["end"]
        lines = []
        for s in range(last + 1):
            objects, people = aggregate(v["secs"].get(s, []))
            lines.append({"video": vid, "second": s, "objects": objects, "people": people})
        if err:
            lines.append({"video": vid, "error": err})
        writer.lines(lines)
        del videos[vid]
        n_videos += 1
        n_secs += last + 1
        if n_videos % 20 == 0:
            el = time.time() - t0
            log(f"{n_videos} videos, {n_secs} s of video, {n_frames} frames inferred, "
                f"{el:.0f}s elapsed, {n_secs / el:.1f} video-s/s, gpu {tm['gpu']:.0f}s, wait {tm['wait']:.0f}s")

    def run_bucket(shape):
        nonlocal n_frames
        items = buckets.pop(shape)
        t = time.time()
        results = det([img for _, _, img in items])
        tm["gpu"] += time.time() - t
        n_frames += len(items)
        touched = set()
        for (vid, sec, _), res in zip(items, results):
            v = videos.get(vid)
            if v is None:
                continue
            v["secs"][sec].append(res)
            v["inflight"] -= 1
            touched.add(vid)
        for vid in touched:
            maybe_finish(vid)

    while wdone < args.workers or buckets:
        t = time.time()
        try:
            msg = out_q.get(timeout=0.05)
        except queue.Empty:
            msg = None
        tm["wait"] += time.time() - t
        if msg is None:
            if buckets:  # GPU idle: run the biggest partial batch
                run_bucket(max(buckets, key=lambda s: len(buckets[s])))
        elif msg[0] == "frames":
            _, vid, sec, _n, imgs = msg
            v = videos.get(vid)
            if v is not None:
                v["inflight"] += len(imgs)
                for img in imgs:
                    b = buckets[img.shape]
                    b.append((vid, sec, img))
                    if len(b) >= args.batch:
                        run_bucket(img.shape)
        elif msg[0] == "start":
            _, wid, vid = msg
            current[wid] = vid
            videos[vid] = {"secs": defaultdict(list), "inflight": 0, "end": None}
        elif msg[0] == "end":
            _, wid, vid, last, err = msg
            current.pop(wid, None)
            if vid in videos:
                videos[vid]["end"] = (last, err)
                maybe_finish(vid)
        elif msg[0] == "wdone":
            wdone += 1

        if time.time() - last_check > 2:  # replace workers killed by a crashing decoder
            last_check = time.time()
            for wid, p in list(procs.items()):
                if not p.is_alive() and p.exitcode not in (0, None):
                    vid = current.pop(wid, None)
                    log(f"worker {wid} died (exit {p.exitcode}) on {vid}; restarting")
                    if vid in videos:
                        secs = videos[vid]["secs"]
                        videos[vid]["end"] = (max(secs) if secs else -1, "decoder crashed")
                        maybe_finish(vid)
                    procs[wid] = spawn(wid)

    for vid in list(videos):  # anything left (should not happen) gets an error line
        v = videos[vid]
        v["inflight"] = 0
        if v["end"] is None:
            v["end"] = (max(v["secs"]) if v["secs"] else -1, "incomplete")
        maybe_finish(vid)
    writer.close()
    el = time.time() - t0
    log(f"DONE: {n_videos} videos, {n_secs} s of video, {n_frames} frames in {el:.1f}s "
        f"(model load {t_ready - t0:.1f}s, gpu {tm['gpu']:.1f}s, waiting on decoders {tm['wait']:.1f}s)")


if __name__ == "__main__":
    main()
