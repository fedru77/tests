"""정확한 분수(Fraction) 행렬 계산: det, REF, RREF, rank.

부동소수를 쓰지 않으므로 1/3 같은 값도 오차 없이 계산됩니다.
행 연산 과정(steps)을 함께 기록해 풀이를 보여줄 수 있습니다.

REF는 유일하지 않습니다. 여기서는 다음 규칙을 따릅니다.
    - 왼쪽 열부터, 아직 피벗이 없는 행 중 위에서 첫 번째로 0이 아닌 행을 피벗으로 고름 (필요하면 행 교환)
    - 피벗을 1로 만들고(선행 1), 그 아래 원소를 0으로 만듦
RREF는 REF에서 각 피벗 위쪽 원소까지 0으로 만든 것이며, 유일합니다.
"""
from fractions import Fraction


def _copy(m):
    return [[Fraction(x) for x in row] for row in m]


def fmt_num(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _coef(c):
    """행 연산 표기용 계수: 1이면 생략, 분수는 괄호"""
    if c == 1:
        return ""
    s = fmt_num(c)
    return f"({s})" if "/" in s else s


def fmt_matrix(m, indent="    "):
    if not m:
        return indent + "[ ]"
    cells = [[fmt_num(x) for x in row] for row in m]
    w = max(len(c) for row in cells for c in row)
    return "\n".join(indent + "[ " + "  ".join(c.rjust(w) for c in row) + " ]" for row in cells)


def det(m, steps=None):
    """행렬식. 정사각 행렬이 아니면 ValueError.

    가우스 소거로 위삼각행렬을 만들고 det = (−1)^(행 교환 수) × (대각 원소의 곱)
    """
    n = len(m)
    if any(len(row) != n for row in m):
        raise ValueError(f"{n}x{len(m[0])} 행렬은 정사각 행렬이 아니라 det가 정의되지 않습니다")
    a = _copy(m)
    sign = 1
    for col in range(n):
        piv = next((r for r in range(col, n) if a[r][col] != 0), None)
        if piv is None:
            if steps is not None:
                steps.append(f"{col + 1}열에 피벗이 없음 → det = 0")
            return Fraction(0)
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            sign = -sign
            if steps is not None:
                steps.append(f"R{col + 1} ↔ R{piv + 1}  (부호 반전)")
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            if f != 0:
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
                if steps is not None:
                    steps.append(f"R{r + 1} ← R{r + 1} − {_coef(f)}R{col + 1}")
    result = Fraction(sign)
    for i in range(n):
        result *= a[i][i]
    if steps is not None:
        diag = " × ".join(fmt_num(a[i][i]) for i in range(n))
        steps.append(f"det = {'−' if sign < 0 else ''}({diag}) = {fmt_num(result)}")
    return result


def ref(m, steps=None):
    """행 사다리꼴(REF, 선행 1). (결과 행렬, 피벗 열 목록)"""
    a = _copy(m)
    rows, cols = len(a), len(a[0])
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = next((i for i in range(r, rows) if a[i][c] != 0), None)
        if piv is None:
            continue
        if piv != r:
            a[r], a[piv] = a[piv], a[r]
            if steps is not None:
                steps.append(f"R{r + 1} ↔ R{piv + 1}")
        p = a[r][c]
        if p != 1:
            a[r] = [x / p for x in a[r]]
            if steps is not None:
                steps.append(f"R{r + 1} ← {_coef(1 / p)}R{r + 1}")
        for i in range(r + 1, rows):
            f = a[i][c]
            if f != 0:
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
                if steps is not None:
                    steps.append(f"R{i + 1} ← R{i + 1} − {_coef(f)}R{r + 1}")
        pivots.append(c)
        r += 1
    return a, pivots


def rref(m, steps=None):
    """기약 행 사다리꼴(RREF). (결과 행렬, 피벗 열 목록)"""
    a, pivots = ref(m, steps)
    for r in reversed(range(len(pivots))):
        c = pivots[r]
        for i in range(r):
            f = a[i][c]
            if f != 0:
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
                if steps is not None:
                    steps.append(f"R{i + 1} ← R{i + 1} − {_coef(f)}R{r + 1}")
    return a, pivots
