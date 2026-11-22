"""Score a results file against a reference (every-frame) results file, per brief section 5.

python tools/compare.py --ref ref.jsonl --test fast.jsonl
"""

import argparse
import gzip
import json


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    secs, errors = {}, set()
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                errors.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (set(d["objects"]), d["people"])
    return secs, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    ref, ref_err = load(a.ref)
    test, test_err = load(a.test)

    tp = fp = fn = exact = within1 = missing = 0
    per_obj = {}
    for key, (robj, rp) in ref.items():
        if key not in test:
            missing += 1
            tobj, tp_ = set(), -99
        else:
            tobj, tp_ = test[key]
        tp += len(robj & tobj)
        fp += len(tobj - robj)
        fn += len(robj - tobj)
        for o in robj ^ tobj:
            per_obj[o] = per_obj.get(o, 0) + 1
        exact += tp_ == rp
        within1 += abs(tp_ - rp) <= 1
    extra = [k for k in test if k not in ref]
    for k in extra:
        fp += len(test[k][0])

    n = len(ref)
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    ex, w1 = exact / n, within1 / n
    print(f"seconds in ref: {n}  missing in test: {missing}  extra in test: {len(extra)}")
    print(f"errors: ref {len(ref_err)}  test {len(test_err)}  differing {len(ref_err ^ test_err)}")
    print(f"object F1      : {f1:.4f}  (>= 0.95)  {'PASS' if f1 >= 0.95 else 'FAIL'}   tp={tp} fp={fp} fn={fn}")
    print(f"people exact   : {ex:.4f}  (>= 0.85)  {'PASS' if ex >= 0.85 else 'FAIL'}")
    print(f"people within 1: {w1:.4f}  (>= 0.95)  {'PASS' if w1 >= 0.95 else 'FAIL'}")
    print(f"object disagreements by class: {per_obj}")


if __name__ == "__main__":
    main()
