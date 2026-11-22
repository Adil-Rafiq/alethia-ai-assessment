### 1. Your timings. How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time? (100 words)

I only ran the optimized version (k=3, FP16 and TensorRT engine) on the full dataset because of time constraints and it took 5687.5s. Sampling the frame rates at `k = 3` cut the run time drastically almost by 11x.

### 2. Your pipeline. Describe what your script does, step by step, from opening a Parquet file to writing an output line. (150 words)

`run.py` parses CLI args, sets-up the output writer and queues, spawns decode worker processes and starts a reader thread that streams the Parquet files from the disk and pushes the video ref and bytes into the `task_q`. The workers read from `task_q`, decode the video, apply sampling if configured, and resize them to `imgsz` while preserving the aspect ratio and write back the images and meta to the `out_q`. The main loop reads `out_q` messages, tracks per-video progress, and groups frames into same-shape buckets for batched inference. When a bucket is full or queue goes idle, `run_bucket` runs the detector on a batch and records per-second results. The `maybe_finish` is called on `end` messages and in `run_bucket` that checks for a completed inference on a video and aggregates its results and write it to disk using the `writer` object.

### 3. Where the time goes. Which part of your final version takes the most time? How did you measure this? (150 words)

GPU takes majority of the time. The GPU took the most time in both the run versions. I measured the GPU time by taking a timestamp before and after calling the detector on a batch inference for all the batches and aggregate the total time which came out to be around 80% of the total run time.

### 4. Checking correctness. How did you check that your results stayed correct after you made the script faster?(100 words)

Initially I made a test subset by sampling a few files and a few rows per file from the original Parquet files. I ran the correct/slow version on it and recorded its results for reference. Then with each speed improvement on the test subset, I cross-checked its results with this reference.

### 5. Working with AI. Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it? (100 words)

The AI agent did everything correct given my instructions. There was nothing to change in the codebase.

### 6. Next steps. If you had one more week, what would you change, and why? (100 words)

If I had more time, I will run the slow version on the full dataset without any optimization. This would have given me the chance to thoroughly benchmark the results.
