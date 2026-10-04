"""학습 스크립트.

    python -m matrix_ml.train --hidden 16 --steps 3000 --out matrix_ml/weights_h16.npz

매 스텝마다 합성 데이터 배치를 새로 생성하므로 같은 예제를 두 번 보지 않습니다(과적합 걱정 없음).
최적화는 Adam, 학습률은 코사인 스케줄로 줄입니다.
"""
import argparse
import random
import time

import numpy as np

from . import datagen, evaluate
from . import model as M


def make_batch(rng, size):
    samples = [datagen.sample(rng) for _ in range(size)]
    lengths = [len(s["ids"]) for s in samples]
    T = max(lengths)
    X = np.zeros((size, T), dtype=np.int64)
    Y = np.zeros((size, T), dtype=np.int64)
    for i, s in enumerate(samples):
        X[i, : lengths[i]] = s["ids"]
        Y[i, : lengths[i]] = s["labels"]
    return X, Y, lengths


class Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        # 기울기 폭주 방지: 전체 노름 클리핑
        norm = np.sqrt(sum((g * g).sum() for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in params:
            g = grads[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--out", default="matrix_ml/weights.npz")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    p = M.init_params(args.hidden, seed=args.seed)
    print(f"hidden={args.hidden}  params={M.count_params(p)}", flush=True)
    opt = Adam(p, args.lr)
    val = evaluate.fixed_set(args.eval_size, seed=12345)

    best, t0, run_loss = -1.0, time.time(), None
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * step / args.steps))  # 코사인 감쇠
        X, Y, L = make_batch(rng, args.batch)
        loss, g = M.loss_and_grads(p, X, Y, L)
        opt.step(p, g, lr)
        run_loss = loss if run_loss is None else 0.98 * run_loss + 0.02 * loss
        if step % 100 == 0:
            print(f"step {step:5d}  loss {run_loss:.4f}  lr {lr:.4f}  {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate.evaluate(p, val)
            print("   eval", evaluate.fmt(r), flush=True)
            if r["top1"] >= best:
                best = r["top1"]
                M.save(p, args.out, hidden=args.hidden, step=step)
                print(f"   saved {args.out} (top1={best:.4f})", flush=True)


if __name__ == "__main__":
    main()
