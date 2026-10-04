"""비교용 규칙 기반 파서. 합성 데이터 생성기(datagen.py)의 규약을 사람이 코드로 옮긴 것입니다.

순서
  1. 행 나누기: 숫자를 담은 가장 안쪽 괄호 묶음이 2개 이상이면 묶음마다 한 행.
     아니면 행 구분자(줄바꿈, ;, |, \\\\)로 나눔.
  2. 각 행에서 숫자 토큰을 정규식으로 뽑음 (부호, 소수, 분수 포함).
  3. 규약 적용
     - 여러 행이 모두 "같은 길이의 숫자만 있는 토큰 하나"면 붙여쓰기 → 한 자리씩 (123/456 → 2×3)
     - 한 행에 같은 길이(2 이상)의 숫자 토큰만 여러 개면 붙여쓰기 행들 (123 453 623 → 3×3)
     - 한 행에 숫자 하나뿐이고 길이가 n²(≥4)이면 전부 붙여쓰기 → n×n (1234 → 2×2)
     - 한 행의 토큰 수가 n²(≥4)이면 n×n (1 2 3 4 → 2×2)
     - 나머지는 보이는 그대로
  각 단계에서 다른 해석(1×N, 열벡터 등)을 뒤 순위 후보로 덧붙입니다.
"""
import re
from fractions import Fraction

_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:/\d+)?")
_ROW_SEP = re.compile(r"\n|;|\||\\\\")
_INNER_GROUP = re.compile(r"[\[\(\{]([^\[\]\(\)\{\}]*)[\]\)\}]")


def _to_frac(s):
    if "/" in s:
        a, b = s.split("/")
        if int(b) == 0:
            raise ValueError(s)
        return Fraction(Fraction(a), int(b))
    return Fraction(s)


def _square_side(n):
    k = int(round(n ** 0.5))
    return k if k >= 2 and k * k == n else None


def _rows(text):
    groups = [g for g in _INNER_GROUP.findall(text) if _NUM.search(g)]
    if len(groups) >= 2:
        return groups
    if len(groups) == 1:
        text = groups[0]
    return [r for r in _ROW_SEP.split(text) if _NUM.search(r)]


def parse(text, top_k=3):
    """규칙 파서. 후보 행렬(문자열 원소) 리스트를 우선순위 순으로 반환"""
    text = text.replace("−", "-").replace("–", "-")
    rows = [_NUM.findall(r) for r in _rows(text)]
    rows = [r for r in rows if r]
    if not rows:
        return []
    cands = []
    digits_only = all(t.isdigit() for r in rows for t in r)
    flat = [t for r in rows for t in r]

    if len(rows) >= 2:
        if digits_only and all(len(r) == 1 for r in rows) and len({len(r[0]) for r in rows}) == 1 \
                and len(rows[0][0]) >= 2:
            cands.append([list(r[0]) for r in rows])     # 123\n456 → 붙여쓰기
            cands.append([[r[0]] for r in rows])          # 열벡터
        if len({len(r) for r in rows}) == 1:
            cands.append(rows)
        n = _square_side(len(flat))
        if n:
            cands.append([flat[i * n:(i + 1) * n] for i in range(n)])
        cands.append([flat])
    else:
        toks = rows[0]
        if len(toks) == 1 and digits_only and _square_side(len(toks[0])):
            n = _square_side(len(toks[0]))
            cands.append([list(toks[0][i * n:(i + 1) * n]) for i in range(n)])  # 1234 → 2×2
        if len(toks) >= 2 and digits_only and len({len(t) for t in toks}) == 1 and len(toks[0]) >= 2:
            cands.append([list(t) for t in toks])        # 123 453 623 → 붙여쓰기 행들
        n = _square_side(len(toks))
        if n:
            cands.append([toks[i * n:(i + 1) * n] for i in range(n)])
        cands.append([toks])
        if len(toks) >= 2:
            cands.append([[t] for t in toks])

    out, seen = [], set()
    for m in cands:
        key = tuple(map(tuple, m))
        if key in seen:
            continue
        try:
            [[_to_frac(x) for x in r] for r in m]
        except (ValueError, ZeroDivisionError):
            continue
        seen.add(key)
        out.append(m)
    return out[:top_k]
