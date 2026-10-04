"""numpy만으로 구현한 초소형 양방향 GRU(BiGRU) 문자 라벨러.

구조
    입력 x_t  : 문자 클래스 one-hot (K=15)
    정방향 GRU: 왼쪽 → 오른쪽으로 읽으며 은닉 상태 h_t^f (H차원)
    역방향 GRU: 오른쪽 → 왼쪽으로 읽으며 은닉 상태 h_t^b (H차원)
    출력      : logits_t = [h_t^f ; h_t^b] · Wo + bo  (4개 라벨 O/I/B/R)

GRU 한 스텝 (PyTorch와 같은 식, 단 리셋 게이트 쪽 bias는 하나로 합침)
    z = σ(Wz x + Uz h' + bz)            업데이트 게이트: 과거를 얼마나 유지할지
    r = σ(Wr x + Ur h' + br)            리셋 게이트: 새 후보를 만들 때 과거를 얼마나 볼지
    n = tanh(Wn x + bn + r ⊙ (Un h'))   새 후보 상태
    h = (1 − z) ⊙ n + z ⊙ h'

x가 one-hot이므로 W·x 는 행렬곱 대신 "W의 열 하나를 꺼내는 것"(임베딩 조회)과 같습니다.
그래서 별도 임베딩 층이 필요 없습니다.

파라미터 수 (방향당): 3H·K + 3H·H + 3H,   출력층: 2H·4 + 4
    H=16 → 2·(720 + 768 + 48) + 132 = 3,204개
"""
import numpy as np

from .charset import LABELS, NUM_CLASSES

NUM_LABELS = len(LABELS)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_params(hidden, seed=0):
    rng = np.random.default_rng(seed)
    K, H = NUM_CLASSES, hidden
    p = {}
    for d in ("f", "b"):  # forward / backward 방향
        s = 1.0 / np.sqrt(H)
        p[f"W{d}"] = rng.uniform(-s, s, (K, 3 * H))  # 입력 → [z | r | n]
        p[f"U{d}"] = rng.uniform(-s, s, (H, 3 * H))  # 은닉 → [z | r | n]
        p[f"b{d}"] = np.zeros(3 * H)
    s = 1.0 / np.sqrt(2 * H)
    p["Wo"] = rng.uniform(-s, s, (2 * H, NUM_LABELS))
    p["bo"] = np.zeros(NUM_LABELS)
    return p


def count_params(p):
    return int(sum(v.size for v in p.values()))


# ---------------------------------------------------------------- GRU 한 방향
def _gru_forward(X, W, U, b):
    """X: (B, T) 클래스 id. 반환: H (B, T, H), 역전파용 캐시"""
    Bsz, T = X.shape
    Hd = U.shape[0]
    h = np.zeros((Bsz, Hd))
    hs = np.zeros((Bsz, T, Hd))
    cache = []
    for t in range(T):
        xw = W[X[:, t]] + b                     # (B, 3H)  one-hot × W = 열 조회
        uh = h @ U                              # (B, 3H)
        z = _sigmoid(xw[:, :Hd] + uh[:, :Hd])
        r = _sigmoid(xw[:, Hd:2 * Hd] + uh[:, Hd:2 * Hd])
        un = uh[:, 2 * Hd:]
        n = np.tanh(xw[:, 2 * Hd:] + r * un)
        h_new = (1 - z) * n + z * h
        cache.append((h, z, r, n, un))
        h = h_new
        hs[:, t] = h
    return hs, cache


def _gru_backward(X, U, cache, dHs):
    """BPTT. dHs: (B, T, H) 각 시점 은닉 상태에 대한 손실의 기울기"""
    Bsz, T, Hd = dHs.shape
    dW = np.zeros((NUM_CLASSES, 3 * Hd))
    dU = np.zeros_like(U)
    db = np.zeros(3 * Hd)
    dh_next = np.zeros((Bsz, Hd))
    for t in reversed(range(T)):
        hp, z, r, n, un = cache[t]
        dh = dHs[:, t] + dh_next
        dn = dh * (1 - z)
        dz = dh * (hp - n)                     # ∂h/∂z = h' − n
        dhp = dh * z
        dan = dn * (1 - n * n)                 # tanh'
        dr = dan * un
        daz = dz * z * (1 - z)                 # σ'
        dar = dr * r * (1 - r)
        dxw = np.concatenate([daz, dar, dan], axis=1)        # 입력 쪽 pre-activation 기울기
        duh = np.concatenate([daz, dar, dan * r], axis=1)    # 은닉 쪽 pre-activation 기울기
        np.add.at(dW, X[:, t], dxw)
        db += dxw.sum(0)
        dU += hp.T @ duh
        dhp += duh @ U.T
        dh_next = dhp
    return dW, dU, db


def _reverse_within_length(X, lengths):
    """각 시퀀스를 자기 길이 안에서만 뒤집음 (패딩은 계속 뒤쪽에 남도록)"""
    Xr = X.copy()
    for i, L in enumerate(lengths):
        Xr[i, :L] = X[i, :L][::-1]
    return Xr


# ---------------------------------------------------------------- 전체 모델
def forward(p, X, lengths):
    """X: (B, T) 클래스 id, lengths: 각 시퀀스 길이. 반환: logits (B, T, 4), 캐시"""
    Hf, cf = _gru_forward(X, p["Wf"], p["Uf"], p["bf"])
    Xr = _reverse_within_length(X, lengths)
    Hb_r, cb = _gru_forward(Xr, p["Wb"], p["Ub"], p["bb"])
    Hb = _reverse_within_length(Hb_r, lengths)
    Hcat = np.concatenate([Hf, Hb], axis=2)              # (B, T, 2H)
    logits = Hcat @ p["Wo"] + p["bo"]
    return logits, (X, Xr, lengths, cf, cb, Hcat)


def log_softmax(logits):
    m = logits.max(-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))


def loss_and_grads(p, X, Y, lengths):
    """문자별 cross-entropy(패딩 제외 평균)와 모든 파라미터의 기울기"""
    logits, (X, Xr, lengths, cf, cb, Hcat) = forward(p, X, lengths)
    Bsz, T, _ = logits.shape
    mask = (np.arange(T)[None, :] < np.asarray(lengths)[:, None]).astype(float)
    logp = log_softmax(logits)
    nll = -np.take_along_axis(logp, Y[..., None], -1)[..., 0]
    ntok = mask.sum()
    loss = (nll * mask).sum() / ntok

    # softmax + CE 의 기울기 = p − onehot(y)
    dlogits = np.exp(logp)
    np.put_along_axis(dlogits, Y[..., None], np.take_along_axis(dlogits, Y[..., None], -1) - 1, -1)
    dlogits *= mask[..., None] / ntok

    g = {}
    g["Wo"] = Hcat.reshape(-1, Hcat.shape[-1]).T @ dlogits.reshape(-1, NUM_LABELS)
    g["bo"] = dlogits.sum((0, 1))
    dHcat = dlogits @ p["Wo"].T
    Hd = p["Uf"].shape[0]
    g["Wf"], g["Uf"], g["bf"] = _gru_backward(X, p["Uf"], cf, dHcat[..., :Hd])
    dHb_r = _reverse_within_length(dHcat[..., Hd:], lengths)
    g["Wb"], g["Ub"], g["bb"] = _gru_backward(Xr, p["Ub"], cb, dHb_r)
    return loss, g


def predict_logprobs(p, ids):
    """시퀀스 하나에 대한 문자별 로그확률 (T, 4)"""
    X = np.asarray(ids, dtype=np.int64)[None, :]
    logits, _ = forward(p, X, [len(ids)])
    return log_softmax(logits[0])


def save(p, path, **meta):
    np.savez_compressed(path, **p, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})


def load(path):
    data = np.load(path)
    return {k: data[k] for k in data.files if not k.startswith("meta_")}
