"""제약 디코딩: 모델이 낸 문자별 확률에서 "말이 되는 행렬"이 되는 라벨열만 골라냅니다.

모델은 문자마다 P(O), P(I), P(B), P(R)을 낼 뿐이고, 그 argmax를 그대로 쓰면
행마다 원소 수가 다르거나(3,3,2), 숫자 중간이 끊기는 결과가 나올 수 있습니다.
그래서 다음 하드 제약을 만족하는 라벨열 중 로그확률 합이 최대인 것을 동적계획법(Viterbi)으로 찾습니다.

  [문자 제약]  직전 글자의 "모드"를 상태로 두고, 상태 전이표로 표현합니다.
    모드: OUT(숫자 밖) / ROW(숫자 밖, 행 구분자를 지나옴) / FIRST(원소를 시작한 숫자 한 글자)
          NUM(여러 글자 숫자 안) / SPLIT(붙여 쓴 한 자리 숫자들)
    - 숫자:    OUT/ROW→B/R→FIRST,  FIRST→I→NUM,  NUM→I→NUM,  FIRST/SPLIT→B/R→SPLIT
               즉 숫자 덩어리 하나는 "통째로 한 숫자"(401)이거나 "전부 한 자리씩"(4,0,1)이고,
               "40"과 "1"처럼 섞어 자르는 해석은 금지
    - 부호(- +): O 또는 B/R→NUM (부호는 원소를 시작함)
    - 소수점:  O,  FIRST/NUM→I→NUM,  OUT/ROW→B/R→NUM (".5")
    - 분수선:  O,  FIRST/NUM→I→NUM
    - 나머지:  O.  행 구분자(줄바꿈 ; | \ 괄호)면 ROW로, 아니면 OUT으로 (ROW는 다음 숫자까지 유지)
  [전역 규약]  한 입력 안에서 두 규약을 섞지 않습니다.
    - 숫자 모드: 모든 숫자 덩어리가 각각 한 숫자 (SPLIT 금지)        예) 12 34 56 78 → 12, 34, ...
    - 분할 모드: 모든 원소가 한 자리 숫자 (I 금지, 붙여쓰기 입력)     예) 123 453 623 → 1, 2, 3, ...
    그래서 각 모양마다 두 모드로 한 번씩 디코딩합니다. "12 34 5 6 78" 같은 섞인 해석이 사라집니다.
    분할 모드에서 1행 해석은 n×n으로 접힐 때만 허용합니다 ("453"을 [4,5,3]으로 자르지 않음).
    숫자 모드에서 R(새 행)은 ROW 모드에서만 가능합니다. 공백/쉼표만으로는 행이 바뀌지 않으므로
    "401 82 973 341"이 4×1 열벡터로 읽히지 않습니다. (분할 모드의 "123 453"은 공백이 행 구분)
  [모양 제약]  행렬 크기 (r, c)를 정해 두고
    - 원소를 시작(B/R)할 때마다 원소 번호 e가 1 증가
    - R 은 e % c == 0 일 때만,  B 는 e % c != 0 일 때만
    - 끝에서 e == r·c
    - (옵션) r == c  정사각 행렬만

가능한 (r, c)를 모두 시도해 각각의 최적 해와 점수를 얻고, 점수 순으로 후보를 정렬합니다.
후보 점수를 softmax 하면 "이 해석이 맞을 확률"(확신도)이 됩니다.

  [행 구분자가 없는 입력]  "1 2 3 4", "1234" 처럼 행 경계가 눈에 보이지 않으면
    모델은 R을 첫 원소에만 붙이도록 학습되어 있어 1×N 해석이 나옵니다.
    N = n² 이면 같은 라벨열을 n×n으로 접은 후보도 만들고, 두 후보에
    모양 사전확률 log(prior), log(1 − prior)를 더해 순위를 정합니다.
    "몇 번째 원소에서 행이 바뀌는가"는 세기 문제라 작은 RNN이 약하므로
    모델이 아니라 디코더가 규칙으로 처리합니다.
"""
import re
from fractions import Fraction

import numpy as np

from .charset import CLS, B, I, O, R

NEG = -1e18

_DIGIT = CLS["DIGIT"]
_SIGN = {CLS["MINUS"], CLS["PLUS"]}
_INNER = {CLS["DOT"], CLS["SLASH"]}

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)(/\d+)?$")


# 직전 글자의 모드
OUT, FIRST, NUM, SPLIT, ROW = 0, 1, 2, 3, 4
NUM_MODES = 5
_ANY = (OUT, FIRST, NUM, SPLIT, ROW)
_GAP = (OUT, ROW)
_ROW_SEP = {CLS[k] for k in ("NEWLINE", "SEMI", "PIPE", "BACKSLASH", "LBRACK", "RBRACK")}


def _sep(cls_id):
    """구분자 O 전이: 행 구분자면 ROW로, 아니면 OUT으로. 이미 ROW면 유지"""
    to = ROW if cls_id in _ROW_SEP else OUT
    return [(m, O, ROW if m == ROW else to) for m in _ANY]


def transitions(cls_id):
    """문자 클래스별 허용 전이 목록: (이전 모드, 라벨, 다음 모드)"""
    if cls_id == _DIGIT:
        return ([(m, lab, FIRST) for m in _GAP for lab in (B, R)]
                + [(m, lab, SPLIT) for m in (FIRST, SPLIT) for lab in (B, R)]
                + [(FIRST, I, NUM), (NUM, I, NUM)])
    if cls_id in _SIGN:
        return _sep(cls_id) + [(m, lab, NUM) for m in _ANY for lab in (B, R)]
    if cls_id == CLS["DOT"]:
        return (_sep(cls_id) + [(FIRST, I, NUM), (NUM, I, NUM)]
                + [(m, lab, NUM) for m in _GAP for lab in (B, R)])
    if cls_id == CLS["SLASH"]:
        return _sep(cls_id) + [(FIRST, I, NUM), (NUM, I, NUM)]
    return _sep(cls_id)


_ALLOWED = {}


def _allowed(cls_id, split_mode):
    """전역 규약에 맞는 전이만 남김 (결과는 캐시)"""
    key = (cls_id, split_mode)
    if key not in _ALLOWED:
        trs = transitions(cls_id)
        if split_mode:
            trs = [tr for tr in trs if tr[1] != I]
        else:
            trs = [tr for tr in trs if tr[2] != SPLIT and (tr[1] != R or tr[0] == ROW)]
        _ALLOWED[key] = trs
    return _ALLOWED[key]


def _viterbi_shape(logp, ids, rows, cols, split_mode=False):
    """주어진 (rows, cols)에서 제약을 만족하는 최고 점수 라벨열. 불가능하면 None.

    상태 = (모드 5종, e: 지금까지 시작한 원소 수 0..N)
    numpy 벡터 연산으로 e 축 전체를 한 번에 갱신합니다.
    """
    N = rows * cols
    T = len(ids)
    e = np.arange(N + 1)
    start_ok = {R: (e % cols == 0) & (e < N),   # 원소 e를 R로 시작할 수 있나
                B: (e % cols != 0) & (e < N)}

    score = np.full((NUM_MODES, N + 1), NEG)
    score[ROW, 0] = 0.0  # 입력의 시작은 행의 시작
    # 역추적용: 각 시점, 각 (모드, e)에 대해 (이전 모드, 라벨)
    bp = np.zeros((T, NUM_MODES, N + 1, 2), dtype=np.int8)

    for t in range(T):
        lp = logp[t]
        new = np.full((NUM_MODES, N + 1), NEG)
        for frm, lab, to in _allowed(ids[t], split_mode):
            cand = score[frm] + lp[lab]
            if lab in (B, R):  # 원소 시작: e → e+1
                src = np.where(start_ok[lab], cand, NEG)
                cand = np.full(N + 1, NEG)
                cand[1:] = src[:-1]
            better = cand > new[to]
            new[to] = np.where(better, cand, new[to])
            bp[t, to, better] = (frm, lab)
        score = new

    best_mode = int(score[:, N].argmax())
    if score[best_mode, N] <= NEG / 2:
        return None
    # 역추적
    labels = [0] * T
    mode, state_e = best_mode, N
    for t in reversed(range(T)):
        frm, lab = bp[t, mode, state_e]
        labels[t] = int(lab)
        if lab in (B, R):
            state_e -= 1
        mode = int(frm)
    return float(score[best_mode, N]), labels


def labels_to_matrix(chars, labels):
    """라벨열 → 문자열 원소들의 2차원 리스트"""
    rows, cur = [], None
    for ch, lab in zip(chars, labels):
        if lab == R:
            rows.append([ch])
            cur = rows[-1]
        elif lab == B:
            cur.append(ch)
        elif lab == I:
            cur[-1] += ch
    return rows


def to_number(s):
    if not _NUMBER_RE.match(s):
        raise ValueError(s)
    if "/" in s:
        num, den = s.split("/")
        if int(den) == 0:
            raise ValueError(s)
        return Fraction(Fraction(num), int(den))
    return Fraction(s)


def candidate_shapes(ids, max_dim=8):
    """시도할 (r, c) 목록. 원소 수 N = r·c 의 범위를 입력에서 미리 좁힙니다.
       하한: 숫자 덩어리(숫자·소수점·분수선이 이어진 구간 중 숫자를 포함한 것) 개수
       상한: 원소를 시작할 수 있는 글자 수"""
    numeric = {_DIGIT} | _INNER
    runs, in_run, has_digit = 0, False, False
    for c in ids + [None]:
        if c in numeric:
            has_digit = has_digit or c == _DIGIT
            in_run = True
        else:
            runs += in_run and has_digit
            in_run = has_digit = False
    starts = sum(1 for c in ids if c == _DIGIT or c in _SIGN or c == CLS["DOT"])
    shapes = [(r, c) for r in range(2, max_dim + 1) for c in range(1, max_dim + 1)
              if runs <= r * c <= starts]
    # 1×N 은 "행 구분자 없음" 해석. n×n(n ≤ max_dim)으로 접힐 수 있으니 N은 max_dim²까지
    shapes += [(1, c) for c in range(1, max_dim * max_dim + 1) if runs <= c <= starts]
    return shapes


def _square_side(n):
    k = int(round(n ** 0.5))
    return k if k >= 2 and k * k == n else None


def fold_square(row):
    """1×n² 행 → n×n 행렬 (행 우선 순서)"""
    n = _square_side(len(row))
    return [row[i * n:(i + 1) * n] for i in range(n)]


def decode(logp, chars, ids, square=False, max_dim=8, top_k=3, flat_square_prior=0.75):
    """후보 행렬들을 점수 순으로 반환. 각 후보: dict(matrix, shape, logprob, confidence, labels)

    flat_square_prior: 행 구분자 없이 n² 개가 나열된 입력을 n×n으로 볼 사전확률
    """
    # 두 자리 이상 이어진 숫자 덩어리가 없으면 두 모드의 결과가 같으므로 숫자 모드만
    has_multi = any(a == _DIGIT and b == _DIGIT for a, b in zip(ids, ids[1:]))
    modes = (False, True) if has_multi else (False,)
    cands = []
    for r, c in candidate_shapes(ids, max_dim):
        if square and r != c and not (r == 1 and _square_side(c)):
            continue
        for split_mode in modes:
            res = _viterbi_shape(logp, ids, r, c, split_mode)
            if res is None:
                continue
            score, labels = res
            m = labels_to_matrix(chars, labels)
            try:
                [[to_number(x) for x in row] for row in m]
            except ValueError:
                continue  # "1/" 처럼 숫자로 읽을 수 없는 원소가 생기면 버림
            n = _square_side(c) if r == 1 else None
            if split_mode and r == 1 and not n:
                continue  # 붙여쓰기는 2행 이상이거나 n×n으로 접힐 때만 ("453"을 4,5,3으로 자르지 않음)
            if n and split_mode:
                # 붙여쓴 1×n²는 n×n으로만 해석 (다른 해석이 없으니 prior도 불필요)
                cands.append({"matrix": fold_square(m[0]), "shape": (n, n),
                              "logprob": score, "labels": labels})
                continue
            if n:
                cands.append({"matrix": fold_square(m[0]), "shape": (n, n),
                              "logprob": score + np.log(flat_square_prior), "labels": labels})
                score += np.log(1 - flat_square_prior)
                if square:
                    continue
            elif r == 1 and c > max_dim:
                continue  # 접을 수 없는 긴 행 벡터는 후보에서 제외
            cands.append({"matrix": m, "shape": (r, c), "logprob": score, "labels": labels})
    if not cands:
        return []
    scores = np.array([d["logprob"] for d in cands])
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    # 서로 다른 라벨열이 같은 행렬을 만들 수 있음 (예: 줄바꿈으로 읽은 3×3과 나열을 접은 3×3).
    # 같은 사건이므로 확률을 더하고, 대표 라벨열은 점수가 가장 높은 것을 씀
    merged = {}
    for d, pr in sorted(zip(cands, probs), key=lambda x: -x[1]):
        key = tuple(map(tuple, d["matrix"]))
        if key in merged:
            merged[key]["confidence"] += float(pr)
        else:
            d["confidence"] = float(pr)
            merged[key] = d
    out = sorted(merged.values(), key=lambda d: -d["confidence"])
    return out[:top_k]


def greedy(logp, chars):
    """비교용: 제약 없이 문자마다 argmax. 행 길이가 안 맞을 수 있음"""
    labels = logp.argmax(-1).tolist()
    # 첫 원소가 B로 시작하면 행이 없으므로 R로 취급
    for t, lab in enumerate(labels):
        if lab in (B, R):
            labels[t] = R
            break
        if lab == I:
            labels[t] = O
    m = labels_to_matrix(chars, labels)
    if len(m) == 1 and _square_side(len(m[0])):
        m = fold_square(m[0])  # 디코더의 prior(0.75 > 0.5)와 같은 선택
    return m
