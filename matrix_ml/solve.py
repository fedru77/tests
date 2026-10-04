"""자유 형식 입력 → 행렬 해석 → det / REF / RREF 계산.

해석이 모호하면(확신도가 --min-conf 이상인 후보가 여럿이면) 후보마다 모두 계산해서 보여줍니다.

    python -m matrix_ml.solve "123 453 623"
    python -m matrix_ml.solve --steps "1 2; 3 4"       # 행 연산 과정까지
    python -m matrix_ml.solve                          # 대화형: 여러 줄 입력 후 빈 줄로 계산
"""
import argparse
import sys

from . import linalg
from .infer import DEFAULT_WEIGHTS, parse_matrix


def candidates(text, min_conf=0.05, top=5, square=False, weights=DEFAULT_WEIGHTS):
    """확신도가 min_conf 이상인 후보들 (최소 1개는 항상 포함)"""
    cands = parse_matrix(text, square=square, top_k=top, weights=weights)
    keep = [c for c in cands if c["confidence"] >= min_conf]
    return keep or cands[:1]


def report(m, show_steps=False):
    """행렬 하나에 대한 det / REF / RREF / rank 결과 문자열"""
    out = []
    rows, cols = len(m), len(m[0])

    steps = [] if show_steps else None
    if rows == cols:
        d = linalg.det(m, steps)
        out.append(f"  det = {linalg.fmt_num(d)}" + ("   (가역)" if d != 0 else "   (비가역)"))
        if show_steps:
            out += ["      " + s for s in steps]
    else:
        out.append(f"  det: {rows}x{cols} 행렬은 정사각 행렬이 아니라 정의되지 않음")

    steps = [] if show_steps else None
    r, pivots = linalg.ref(m, steps)
    out.append("  REF:")
    if show_steps:
        out += ["      " + s for s in steps] or ["      (행 연산 없음)"]
    out.append(linalg.fmt_matrix(r))

    steps = [] if show_steps else None
    rr, pivots = linalg.rref(m, steps)
    out.append("  RREF:")
    if show_steps:
        out += ["      " + s for s in steps] or ["      (행 연산 없음)"]
    out.append(linalg.fmt_matrix(rr))
    out.append(f"  rank = {len(pivots)}   피벗 열: {', '.join(str(c + 1) for c in pivots) or '없음'}")
    return "\n".join(out)


def solve_text(text, min_conf=0.05, top=5, square=False, show_steps=False, weights=DEFAULT_WEIGHTS):
    cands = candidates(text, min_conf, top, square, weights)
    if not cands:
        return "해석 실패: 숫자를 찾을 수 없습니다."
    out = []
    if len(cands) > 1:
        out.append(f"입력이 모호합니다. 가능한 해석 {len(cands)}개를 모두 계산합니다.")
    for k, c in enumerate(cands, 1):
        r, cc = c["shape"]
        if out:
            out.append("")
        out.append(f"=== 후보 {k}: {r}x{cc} 행렬 (확신도 {c['confidence']:.1%}) ===")
        out.append(linalg.fmt_matrix(c["matrix"]))
        out.append(report(c["matrix"], show_steps))
    return "\n".join(out)


def _interactive(args):
    print("행렬을 입력하세요. 여러 줄이면 줄마다 입력하고, 빈 줄을 입력하면 계산합니다. (종료: Ctrl+C 또는 q)")
    while True:
        lines = []
        try:
            while True:
                line = input("> " if not lines else "  ")
                if line.strip().lower() in ("q", "quit", "exit") and not lines:
                    return
                if not line.strip():
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print()
            if lines:
                print(solve_text("\n".join(lines), args.min_conf, args.top, args.square, args.steps, args.weights))
            return
        if lines:
            print(solve_text("\n".join(lines), args.min_conf, args.top, args.square, args.steps, args.weights))
            print()


def main():
    ap = argparse.ArgumentParser(description="자유 형식 행렬 입력의 det / REF / RREF 계산")
    ap.add_argument("text", nargs="?", help="행렬 문자열. 생략하면 대화형 모드")
    ap.add_argument("--steps", action="store_true", help="행 연산 과정도 출력")
    ap.add_argument("--min-conf", type=float, default=0.05,
                    help="이 확신도 이상인 후보는 모두 계산 (기본 0.05 = 5%%)")
    ap.add_argument("--top", type=int, default=5, help="최대 후보 수 (기본 5)")
    ap.add_argument("--square", action="store_true", help="정사각 행렬 후보만")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = ap.parse_args()

    try:  # Windows 콘솔 등에서 한글/기호 출력이 깨지지 않도록
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.text is None:
        _interactive(args)
        return
    text = args.text.replace("\\n", "\n")  # 명령줄에서 "1 2\n3 4"처럼 줄바꿈을 \n으로 쓸 수 있게
    print(solve_text(text, args.min_conf, args.top, args.square, args.steps, args.weights))


if __name__ == "__main__":
    main()
