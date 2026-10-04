"""사용 API와 CLI.

    from matrix_ml.infer import parse_matrix
    parse_matrix("123 453 623")
    # → [{'matrix': [[1,2,3],[4,5,3],[6,2,3]], 'shape': (3,3), 'confidence': 0.97, ...}, ...]

    python -m matrix_ml.infer "123 453 623"
    python -m matrix_ml.infer --square "1 2 3 4 5 6 7 8 9"
"""
import argparse
import os

from . import decode
from . import model as M
from .charset import LABELS, normalize

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights.npz")
_cache = {}


def _params(path):
    if path not in _cache:
        _cache[path] = M.load(path)
    return _cache[path]


def parse_matrix(text, square=False, top_k=3, weights=DEFAULT_WEIGHTS):
    """자유 형식 문자열 → 행렬 후보 리스트(확신도 내림차순).

    각 후보: matrix(Fraction 2차원 리스트), raw(문자열 원소), shape, confidence, tags(문자별 라벨)
    빈 리스트면 어떤 행렬로도 해석할 수 없다는 뜻입니다.
    """
    chars, ids = normalize(text)
    if not ids:
        return []
    logp = M.predict_logprobs(_params(weights), ids)
    cands = decode.decode(logp, chars, ids, square=square, top_k=top_k)
    for c in cands:
        c["raw"] = c["matrix"]
        c["matrix"] = [[decode.to_number(x) for x in row] for row in c["raw"]]
        c["tags"] = "".join(LABELS[l] for l in c.pop("labels"))
        c["text"] = "".join(chars)
    return cands


def _show(x):
    return str(x)


def main():
    ap = argparse.ArgumentParser(description="자유 형식 행렬 입력 해석기")
    ap.add_argument("text")
    ap.add_argument("--square", action="store_true", help="정사각 행렬만 후보로")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--tags", action="store_true", help="문자별 라벨도 출력")
    args = ap.parse_args()

    text = args.text.encode().decode("unicode_escape") if "\\n" in args.text else args.text
    cands = parse_matrix(text, square=args.square, top_k=args.top, weights=args.weights)
    if not cands:
        print("해석 실패")
        return
    for k, c in enumerate(cands, 1):
        r, cc = c["shape"]
        print(f"[{k}] {r}x{cc}  확신도 {c['confidence']:.1%}")
        if args.tags:
            print("    " + c["text"].replace("\n", "⏎"))
            print("    " + c["tags"])
        w = max(len(_show(x)) for row in c["matrix"] for x in row)
        for row in c["matrix"]:
            print("    [ " + "  ".join(_show(x).rjust(w) for x in row) + " ]")


if __name__ == "__main__":
    main()
