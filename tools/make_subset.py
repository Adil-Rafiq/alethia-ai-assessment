"""Build a small, varied dev set: a few rows from each of many parquet files.

python tools/make_subset.py --input ~/data/finevideo/data --output ~/data/subset --files 20 --rows 2
"""

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--files", type=int, default=20)
    ap.add_argument("--rows", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input, "*.parquet")))
    step = max(1, len(files) // a.files)
    for i, path in enumerate(files[::step][: a.files]):
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        want = {(i * 7 + j * (n // a.rows)) % n for j in range(a.rows)}  # spread rows out
        rows, r = [], 0
        for batch in pf.iter_batches(batch_size=1, columns=["mp4", "json"]):
            if r in want:
                rows.append(batch)
            r += 1
        out = os.path.join(a.output, os.path.basename(path))
        pq.write_table(pa.Table.from_batches(rows), out)
        print(out, sorted(want))


if __name__ == "__main__":
    main()
