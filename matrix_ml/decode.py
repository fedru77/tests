"""제약 디코딩: 모델이 낸 문자별 확률에서 "말이 되는 행렬"이 되는 라벨열만 골라냅니다.

모델은 문자마다 P(O), P(I), P(B), P(R)을 낼 뿐이고, 그 argmax를 그대로 쓰면
행마다 원소 수가 다르거나(3,3,2), 숫자 중간이 끊기는 결과가 나올 수 있습니다.
그래서 다음 하드 제약을 만족하는 라벨열 중 로그확률 합이 최대인 것을 동적계획법(Viterbi)으로 찾습니다.

  [문자 제약]
    - 숫자 문자:       I / B / R 만 가능 (숫자는 반드시 어떤 원소에 속함)
    - 부호(- +):       B / R / O
    - 소수점:          O / I / B / R (".5"처럼 원소를 시작할 수도 있음)
    - 분수선:          I / O
    - 나머지 구분자:   O 만 가능
    - I 는 바로 앞 글자가 숫자의 일부일 때만 가능 (원소는 연속된 글자)
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


def allowed_labels(cls_id):
    """문자 클래스별로 가능한 라벨 마스크 (O, I, B, R)"""
    if cls_id == _DIGIT:
        return (False, True, True, True)
    if cls_id in _SIGN:
        return (True, False, True, True)
    if cls_id == CLS["DOT"]:
        return (True, True, True, True)
    if cls_id == CLS["SLASH"]:
        return (True, True, False, False)
    return (True, False, False, False)


def _viterbi_shape(logp, ids, rows, cols):
    """주어진 (rows, cols)에서 제약을 만족하는 최고 점수 라벨열. 불가능하면 None.

    상태 = (e: 지금까지 시작한 원소 수 0..N,  inside: 직전 글자가 숫자의 일부인가)
    numpy 벡터 연산으로 e 축 전체를 한 번에 갱신합니다.
    """
    N = rows * cols
    T = len(ids)
    e = np.arange(N + 1)
    can_R = (e % cols == 0) & (e < N)   # 원소 e를 R로 시작할 수 있나
    can_B = (e % cols != 0) & (e < N)

    out = np.full(N + 1, NEG)  # inside=False
    inn = np.full(N + 1, NEG)  # inside=True
    out[0] = 0.0
    # 역추적용: 각 시점, 각 (e, inside)에 대해 (이전 inside, 라벨) 저장
    bp_out = np.zeros((T, N + 1), dtype=np.int8)  # 이전 inside (라벨은 항상 O)
    bp_in = np.zeros((T, N + 1, 2), dtype=np.int8)  # (이전 inside, 라벨)

    for t in range(T):
        aO, aI, aB, aR = allowed_labels(ids[t])
        lp = logp[t]
        best_prev = np.maximum(out, inn)
        prev_flag = (inn > out).astype(np.int8)

        new_out = np.full(N + 1, NEG)
        new_in = np.full(N + 1, NEG)
        if aO:
            new_out = best_prev + lp[O]
            bp_out[t] = prev_flag
        if aI:
            cand = inn + lp[I]
            better = cand > new_in
            new_in = np.where(better, cand, new_in)
            bp_in[t, better] = (1, I)
        # 원소 시작: e → e+1
        for lab, ok, mask in ((B, aB, can_B), (R, aR, can_R)):
            if not ok:
                continue
            src = np.where(mask, best_prev + lp[lab], NEG)
            cand = np.full(N + 1, NEG)
            cand[1:] = src[:-1]
            flags = np.zeros(N + 1, dtype=np.int8)
            flags[1:] = prev_flag[:-1]
            better = cand > new_in
            new_in = np.where(better, cand, new_in)
            bp_in[t, better, 0] = flags[better]
            bp_in[t, better, 1] = lab
        out, inn = new_out, new_in

    if max(out[N], inn[N]) <= NEG / 2:
        return None
    # 역추적
    labels = [0] * T
    state_e, inside = N, int(inn[N] > out[N])
    score = max(out[N], inn[N])
    for t in reversed(range(T)):
        if inside:
            prev_inside, lab = bp_in[t, state_e]
            labels[t] = int(lab)
            if lab in (B, R):
                state_e -= 1
            inside = int(prev_inside)
        else:
            labels[t] = O
            inside = int(bp_out[t, state_e])
    return float(score), labels


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
    cands = []
    for r, c in candidate_shapes(ids, max_dim):
        if square and r != c and not (r == 1 and _square_side(c)):
            continue
        res = _viterbi_shape(logp, ids, r, c)
        if res is None:
            continue
        score, labels = res
        m = labels_to_matrix(chars, labels)
        try:
            [[to_number(x) for x in row] for row in m]
        except ValueError:
            continue  # "1/" 처럼 숫자로 읽을 수 없는 원소가 생기면 버림
        n = _square_side(c) if r == 1 else None
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
    cands.sort(key=lambda d: -d["logprob"])
    scores = np.array([d["logprob"] for d in cands])
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    for d, pr in zip(cands, probs):
        d["confidence"] = float(pr)
    return cands[:top_k]


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
