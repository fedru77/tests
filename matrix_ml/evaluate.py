"""평가: 고정된 검증 세트에서 문자 정확도, 제약 없는 argmax 정확도, 제약 디코딩 top-1/top-3 정확도.

    python -m matrix_ml.evaluate matrix_ml/weights.npz --size 2000
"""
import argparse
import random
from collections import defaultdict

from . import datagen, decode
from . import model as M


def fixed_set(size, seed):
    rng = random.Random(seed)
    return [datagen.sample(rng) for _ in range(size)]


def evaluate(p, samples, per_format=False):
    n = len(samples)
    char_ok = char_n = greedy_ok = top1 = top3 = 0
    fmt_stat = defaultdict(lambda: [0, 0])
    for s in samples:
        logp = M.predict_logprobs(p, s["ids"])
        pred = logp.argmax(-1)
        char_ok += int((pred == s["labels"]).sum())
        char_n += len(s["labels"])
        greedy_ok += decode.greedy(logp, s["chars"]) == s["matrix"]
        cands = decode.decode(logp, s["chars"], s["ids"], top_k=3)
        hit1 = bool(cands) and cands[0]["matrix"] == s["matrix"]
        top1 += hit1
        top3 += any(c["matrix"] == s["matrix"] for c in cands)
        fmt_stat[s["format"]][0] += hit1
        fmt_stat[s["format"]][1] += 1
    r = {"char_acc": char_ok / char_n, "greedy": greedy_ok / n, "top1": top1 / n, "top3": top3 / n}
    if per_format:
        r["per_format"] = {k: v[0] / v[1] for k, v in sorted(fmt_stat.items())}
    return r


def fmt(r):
    return (f"char_acc {r['char_acc']:.4f} | 제약없음(argmax) {r['greedy']:.4f} | "
            f"제약디코딩 top1 {r['top1']:.4f} top3 {r['top3']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()
    p = M.load(args.weights)
    print(f"params={M.count_params(p)}")
    r = evaluate(p, fixed_set(args.size, args.seed), per_format=True)
    print(fmt(r))
    for k, v in r["per_format"].items():
        print(f"  {k:12s} top1 {v:.4f}")
