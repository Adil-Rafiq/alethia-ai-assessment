"""Decode worker processes. Must NOT import torch/ultralytics (spawned, kept light)."""

import io
import math
import traceback
from fractions import Fraction

import av
import cv2

from . import IMGSZ

_ROTATE = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def letterbox_resize(img, imgsz=IMGSZ):
    """Exactly the resize step of Ultralytics LetterBox (padding is left to Ultralytics).

    After this, Ultralytics computes r == 1 for the image and skips its own resize, so results
    are bit-identical while we ship ~4x fewer bytes between processes.
    """
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_unpad = (max(round(w * r), 1), max(round(h * r), 1))
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return img


def frame_to_input(frame):
    """VideoFrame -> rotated, resized BGR ndarray."""
    img = frame.to_ndarray(format="bgr24")
    # frame.rotation is the display-matrix angle (counter-clockwise degrees, FFmpeg convention).
    rot = int(round(frame.rotation or 0)) % 360
    if rot in _ROTATE:
        img = cv2.rotate(img, _ROTATE[rot])
    return letterbox_resize(img)


def pick(frames, k):
    """k evenly spaced frames from one second's frames (all of them if k == 0 or n <= k)."""
    n = len(frames)
    if k <= 0 or n <= k:
        return frames
    return [frames[int((j + 0.5) * n / k)] for j in range(k)]


def decode_video(vid, data, k, emit, max_seconds=0):
    """Decode one video, emitting ('frames', vid, sec, n_in_second, [imgs]) per second.

    Returns (last_sec, error_or_None). last_sec is the highest second that had a frame.
    """
    last_sec = -1
    cur_sec, buf = None, []

    def flush():
        if cur_sec is not None and buf:
            imgs = [frame_to_input(f) for f in pick(buf, k)]
            emit(("frames", vid, cur_sec, len(buf), imgs))

    try:
        container = av.open(io.BytesIO(data), metadata_errors="ignore")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        rate = stream.average_rate or stream.guessed_rate or Fraction(30)
        pts0, idx = None, 0
        for frame in container.decode(stream):
            if frame.pts is not None and tb is not None:
                if pts0 is None:
                    pts0 = frame.pts
                t = (frame.pts - pts0) * tb
            else:
                t = Fraction(idx) / Fraction(rate)
            idx += 1
            sec = max(0, math.floor(t))
            if max_seconds and sec >= max_seconds:
                break
            if sec != cur_sec:
                flush()
                cur_sec, buf = sec, []
            buf.append(frame)
            last_sec = max(last_sec, sec)
        flush()
        container.close()
        if last_sec < 0:
            return last_sec, "could not decode video"
        return last_sec, None
    except Exception as e:  # damaged video: keep what we have, report error
        try:
            flush()
        except Exception:
            pass
        msg = f"decode failed: {type(e).__name__}: {str(e)[:120]}"
        return last_sec, msg


def worker_main(wid, task_q, out_q, k, max_seconds):
    cv2.setNumThreads(1)
    while True:
        task = task_q.get()
        if task is None:
            out_q.put(("wdone", wid))
            return
        vid, data = task
        out_q.put(("start", wid, vid))
        try:
            last_sec, err = decode_video(vid, data, k, out_q.put, max_seconds)
        except Exception:
            last_sec, err = -1, "worker error: " + traceback.format_exc(limit=1)[-150:]
        out_q.put(("end", wid, vid, last_sec, err))
