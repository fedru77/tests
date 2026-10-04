"""ML 해석기(BiGRU + 제약 디코더) vs 규칙 파서 비교.

    python -m matrix_ml.compare            # 두 시험 모두
    python -m matrix_ml.compare --size 500 # 분포 안 시험 크기 조절

시험 1. 분포 안: 합성 데이터 생성기로 만든 검증 세트 (생성 규약 = 규칙 파서의 규약이라 규칙 파서에 유리)
시험 2. 분포 밖: 생성기에 없는 쓰기 방식. 사람이 의도한 정답을 직접 적었습니다.
        이 목록은 두 방법의 결과를 보기 전에 고정했습니다.
"""
import argparse
import random
from fractions import Fraction

from . import datagen, rule_parser
from .infer import parse_matrix

# (분류, 입력, 사람이 의도한 행렬)
OOD_CASES = [
    # 구분자 조합이 생성기와 다름
    ("구분자 조합", "1;2;3\n4;5;6", [[1, 2, 3], [4, 5, 6]]),
    ("구분자 조합", "1 2 3, 4 5 6, 7 8 9", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ("구분자 조합", "[1,2,3;4,5,6]", [[1, 2, 3], [4, 5, 6]]),
    ("구분자 조합", "[[1 2] [3 4]]", [[1, 2], [3, 4]]),
    ("구분자 조합", "|1 2|\n|3 4|", [[1, 2], [3, 4]]),
    ("구분자 조합", "1 2 3 / 4 5 6", [[1, 2, 3], [4, 5, 6]]),
    ("구분자 조합", "1,2,3 ; 4,5,6 ; 7,8,9 ;", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ("구분자 조합", "1\t2\r\n3\t4", [[1, 2], [3, 4]]),
    ("구분자 조합", "1 2 3\n\n\n4 5 6", [[1, 2, 3], [4, 5, 6]]),
    ("구분자 조합", "{1, 2; 3, 4}", [[1, 2], [3, 4]]),
    # 앞뒤에 글자가 붙음
    ("글자 섞임", "A = [1 2; 3 4]", [[1, 2], [3, 4]]),
    ("글자 섞임", "det [[2,1],[1,3]]", [[2, 1], [1, 3]]),
    ("글자 섞임", "matrix: 1 2 3 / 4 5 6 / 7 8 10", [[1, 2, 3], [4, 5, 6], [7, 8, 10]]),
    ("글자 섞임", "첫째 행 1 2, 둘째 행 3 4", [[1, 2], [3, 4]]),
    ("글자 섞임", "rref of 1 2 3; 4 5 6", [[1, 2, 3], [4, 5, 6]]),
    ("글자 섞임", "2x2: 5 6 7 8", [[5, 6], [7, 8]]),
    ("글자 섞임", "r1: 1 0 2\nr2: 0 1 3", [[1, 0, 2], [0, 1, 3]]),
    # 큰 행렬, 특이한 숫자
    ("크기/숫자", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
    ("크기/숫자", "1 0 0 0 0 0 0\n0 1 0 0 0 0 0\n0 0 1 0 0 0 0\n0 0 0 1 0 0 0\n0 0 0 0 1 0 0\n0 0 0 0 0 1 0\n0 0 0 0 0 0 1",
     [[int(i == j) for j in range(7)] for i in range(7)]),
    ("크기/숫자", "−1 2; 3 −4", [[-1, 2], [3, -4]]),
    ("크기/숫자", "+1 -1\n-1 +1", [[1, -1], [-1, 1]]),
    ("크기/숫자", "0.5 .25\n-1. 2", [[Fraction(1, 2), Fraction(1, 4)], [-1, 2]]),
    ("크기/숫자", "1/2 -3/4 5\n6 7/8 -9", [[Fraction(1, 2), Fraction(-3, 4), 5], [6, Fraction(7, 8), -9]]),
    ("크기/숫자", "100 200 300\n400 500 600\n700 800 900", [[100, 200, 300], [400, 500, 600], [700, 800, 900]]),
    # 생성기 규약 그대로 (대조군)
    ("대조군", "123 453 623", [[1, 2, 3], [4, 5, 3], [6, 2, 3]]),
    ("대조군", "[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
    ("대조군", "1 2; 3 4", [[1, 2], [3, 4]]),
    ("대조군", "1 -2 3\n4 5 -6\n7 8 9", [[1, -2, 3], [4, 5, -6], [7, 8, 9]]),
    ("대조군", "1234", [[1, 2], [3, 4]]),
    ("대조군", "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}", [[1, 2], [3, 4]]),
]


def _num(m):
    return [[Fraction(x) if not isinstance(x, str) else rule_parser._to_frac(x) for x in r] for r in m]


def ml_cands(text):
    return [c["matrix"] for c in parse_matrix(text, top_k=3)]


def rule_cands(text):
    return [_num(m) for m in rule_parser.parse(text, top_k=3)]


def in_distribution(size, seed=999):
    rng = random.Random(seed)
    stats = {"ML": [0, 0], "규칙": [0, 0]}
    for _ in range(size):
        s = datagen.sample(rng)
        want = _num(s["matrix"])
        for name, fn in (("ML", ml_cands), ("규칙", rule_cands)):
            c = fn(s["text"])
            stats[name][0] += bool(c) and c[0] == want
            stats[name][1] += want in c
    print(f"[시험 1] 분포 안 (합성 검증 세트 {size}개)")
    for name, (a, b) in stats.items():
        print(f"  {name:4s}  top-1 {a / size:6.2%}   top-3 {b / size:6.2%}")


def out_of_distribution(verbose=True):
    print(f"\n[시험 2] 분포 밖 ({len(OOD_CASES)}개, 사람이 정답을 적음)")
    by_cat = {}
    rows = []
    for cat, text, want in OOD_CASES:
        want = _num(want)
        res = {}
        for name, fn in (("ML", ml_cands), ("규칙", rule_cands)):
            c = fn(text)
            res[name] = (bool(c) and c[0] == want, want in c)
        st = by_cat.setdefault(cat, {"n": 0, "ML": [0, 0], "규칙": [0, 0]})
        st["n"] += 1
        for name in ("ML", "규칙"):
            st[name][0] += res[name][0]
            st[name][1] += res[name][1]
        rows.append((cat, text, res))
    if verbose:
        mark = lambda r: "O" if r[0] else ("△" if r[1] else "X")
        print("  (O = 1순위 정답, △ = 3순위 안, X = 실패)")
        print(f"  {'ML':>2} {'규칙':>2}  입력")
        for cat, text, res in rows:
            print(f"  {mark(res['ML']):>2} {mark(res['규칙']):>3}   [{cat}] {text!r}")
    print(f"\n  {'분류':10s} {'개수':>4}   {'ML top-1':>9} {'top-3':>6}   {'규칙 top-1':>9} {'top-3':>6}")
    tot = {"n": 0, "ML": [0, 0], "규칙": [0, 0]}
    for cat, st in by_cat.items():
        print(f"  {cat:10s} {st['n']:>4}   {st['ML'][0]:>9} {st['ML'][1]:>6}   {st['규칙'][0]:>9} {st['규칙'][1]:>6}")
        tot["n"] += st["n"]
        for name in ("ML", "규칙"):
            tot[name][0] += st[name][0]
            tot[name][1] += st[name][1]
    print(f"  {'합계':10s} {tot['n']:>4}   {tot['ML'][0]:>9} {tot['ML'][1]:>6}   {tot['규칙'][0]:>9} {tot['규칙'][1]:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2000)
    args = ap.parse_args()
    in_distribution(args.size)
    out_of_distribution()


if __name__ == "__main__":
    main()
