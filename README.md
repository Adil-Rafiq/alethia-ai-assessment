# Video indexing with YOLO11l

For each second of each video in a folder of FineVideo Parquet files, this records which of
`person, car, dog, laptop, cell phone` are present and how many people are visible.

## Install

No `apt-get` packages are needed. The PyAV wheel bundles FFmpeg.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (run once, needs internet, untimed)

```bash
python prepare.py
```

This downloads `yolo11l.pt` and builds a TensorRT FP16 engine, `yolo11l.engine`, for the local GPU.
The build takes a few minutes. The engine is specific to the GPU it was built on, so it has to be
built on the machine that runs the script.

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

`run.py` uses `yolo11l.engine` if it exists. Otherwise it falls back to `yolo11l.pt`, which is
slower but gives the same results.

## How it works

1. A reader thread streams the `mp4` column one row at a time, so a whole 0.5 GB Parquet file is
   never loaded into memory.
2. Decode worker processes (PyAV) decode each video. Each frame's time is counted from the first
   decoded frame, and the frame goes in second `floor(t)`. Rotation tags are applied. From each
   second, the worker keeps **3 evenly spaced frames** and resizes them exactly as the Ultralytics
   letterbox does.
3. The GPU process batches frames of the same shape from many videos. Grouping by shape keeps the
   same minimal-rectangle padding Ultralytics uses. It runs YOLO11l at an input size of 640 with
   conf 0.4 and the default class-aware NMS (IoU 0.7).
4. For each second, an object is present if it is detected in at least half of the sampled frames.
   The people count is the median, rounded down when it falls halfway between two numbers.
5. Each video's lines are written once its last frame has been processed. Damaged videos, videos
   that fail partway through, crashed decoders and unreadable Parquet files each produce one error
   line, and the run continues.

## Useful options

| flag | meaning |
|---|---|
| `--k N` | frames sampled per second (default 3; `0` = every frame) |
| `--fp32 --weights yolo11l.pt --k 0` | exact every-frame reference run |
| `--workers N` | decode processes (default `min(16, cpu_count-2)`) |
| `--batch N` | inference batch size (default 8; must be no larger than the engine's batch) |
| `--skip-nonref` | don't decode non-reference frames (less decode work) |

## Checking accuracy

```bash
python tools/make_subset.py --input finevideo/data --output subset --files 20 --rows 2
python run.py --input subset --output ref.jsonl --k 0 --fp32 --weights yolo11l.pt --max-seconds 60
python run.py --input subset --output fast.jsonl --max-seconds 60
python tools/compare.py --ref ref.jsonl --test fast.jsonl
```

`compare.py` reports object F1, the share of seconds with an exact people count, and the share
within 1, against the thresholds in the brief.
