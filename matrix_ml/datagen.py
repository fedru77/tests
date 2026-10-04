"""합성 학습 데이터 생성기.

랜덤 행렬을 만들고 → 사람이 칠 법한 여러 형식의 문자열로 바꾸면서 → 문자마다 정답 라벨을 동시에 기록합니다.
정답을 이미 알고 문자열을 만드니 라벨링 비용이 0이고, 데이터를 무한히 만들 수 있습니다.

라벨: O(숫자 아님) / I(숫자 계속) / B(새 원소) / R(새 원소 + 새 행)
"""
import random
from fractions import Fraction

from .charset import B, I, O, R, normalize


# ---------------------------------------------------------------- 값 생성
def _sample_values(rows, cols, rng, style):
    def one():
        if style == "digit":
            return str(rng.randint(0, 9))
        if style == "small":
            return str(rng.randint(-20, 20))
        if style == "big":
            v = rng.randint(-999, 999) if rng.random() < 0.8 else rng.randint(-99999, 99999)
            return str(v)
        if style == "decimal":
            v = round(rng.uniform(-50, 50), rng.choice([1, 1, 2, 3]))
            s = str(v)
            if rng.random() < 0.3:
                s = str(rng.randint(-9, 9))  # 정수가 섞인 실수 행렬
            elif rng.random() < 0.1 and s.startswith("0."):
                s = s[1:]  # ".5"
            return s
        if style == "fraction":
            if rng.random() < 0.4:
                return str(rng.randint(-9, 9))
            f = Fraction(rng.randint(-12, 12), rng.randint(2, 9))
            return str(f)
        raise ValueError(style)

    return [[one() for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------- 문자열 + 라벨 빌더
class _Builder:
    def __init__(self):
        self.text = []
        self.labels = []

    def sep(self, s):
        """구분자/잡음: 전부 O"""
        self.text.extend(s)
        self.labels.extend([O] * len(s))

    def num(self, s, first_in_row):
        """숫자 한 개: 첫 글자는 R 또는 B, 나머지는 I"""
        self.text.extend(s)
        self.labels.append(R if first_in_row else B)
        self.labels.extend([I] * (len(s) - 1))

    def result(self):
        return "".join(self.text), self.labels


def _sp(rng):
    """구분자 주위에 무작위로 들어가는 공백 (사람이 아무렇게나 치는 것 흉내)"""
    r = rng.random()
    if r < 0.55:
        return ""
    if r < 0.9:
        return " "
    return "  "


# ---------------------------------------------------------------- 형식들
def _fmt_delimited(m, rng):
    """행/열 구분자가 명시된 형식: [[1,2],[3,4]], 1 2; 3 4, 줄바꿈, | , LaTeX 등"""
    kind = rng.choice(["nested", "nested", "matlab", "lines", "lines", "pipe", "latex", "rowbrackets"])
    b = _Builder()
    if kind == "nested":  # [[1, 2], [3, 4]]  /  ((1,2),(3,4))
        lo, hi = rng.choice([("[", "]"), ("[", "]"), ("(", ")"), ("{", "}")])
        esep = rng.choice([",", ", ", ",", " , "])
        rsep = rng.choice([",", ", ", ",\n", ",", " "])
        outer = rng.random() < 0.85
        if outer:
            b.sep(lo)
        for i, row in enumerate(m):
            if i:
                b.sep(rsep)
            b.sep(lo + _sp(rng) * (rng.random() < 0.2))
            for j, v in enumerate(row):
                if j:
                    b.sep(esep)
                b.num(v, j == 0)
            b.sep(hi)
        if outer:
            b.sep(hi)
    elif kind == "matlab":  # [1 2; 3 4]  /  1,2;3,4
        esep = rng.choice([" ", ",", ", ", " "])
        rsep = rng.choice([";", "; ", " ; ", ";\n"])
        br = rng.random() < 0.5
        if br:
            b.sep("[" + _sp(rng))
        for i, row in enumerate(m):
            if i:
                b.sep(rsep)
            for j, v in enumerate(row):
                if j:
                    b.sep(esep)
                b.num(v, j == 0)
        if br:
            b.sep(_sp(rng) + "]")
    elif kind == "lines":  # 1 2\n3 4
        esep = rng.choice([" ", "  ", ",", ", ", "\t", " "])
        rsep = rng.choice(["\n", "\n", " \n", "\n\n", ",\n"])
        for i, row in enumerate(m):
            if i:
                b.sep(rsep)
            for j, v in enumerate(row):
                if j:
                    b.sep(esep)
                b.num(v, j == 0)
    elif kind == "pipe":  # 1 2 | 3 4
        esep = rng.choice([" ", ",", ", "])
        rsep = rng.choice([" | ", "|", " |", "| "])
        for i, row in enumerate(m):
            if i:
                b.sep(rsep)
            for j, v in enumerate(row):
                if j:
                    b.sep(esep)
                b.num(v, j == 0)
    elif kind == "latex":  # \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
        env = rng.choice(["bmatrix", "pmatrix", "matrix", None])
        if env:
            b.sep("\\begin{%s}" % env + rng.choice(["", " ", "\n"]))
        for i, row in enumerate(m):
            if i:
                b.sep(rng.choice([" \\\\ ", "\\\\", " \\\\\n", "\\\\ "]))
            for j, v in enumerate(row):
                if j:
                    b.sep(rng.choice([" & ", "&", " &", "& "]))
                b.num(v, j == 0)
        if env:
            b.sep(rng.choice(["", " ", "\n"]) + "\\end{%s}" % env)
    else:  # rowbrackets: [1 2] [3 4]  /  (1,2)(3,4)
        lo, hi = rng.choice([("[", "]"), ("(", ")")])
        esep = rng.choice([" ", ",", ", "])
        rsep = rng.choice([" ", "", ", ", "\n"])
        for i, row in enumerate(m):
            if i:
                b.sep(rsep)
            b.sep(lo)
            for j, v in enumerate(row):
                if j:
                    b.sep(esep)
                b.num(v, j == 0)
            b.sep(hi)
    return b.result()


def _fmt_concat_rows(m, rng):
    """한 자리 숫자를 행마다 붙여 쓴 형식: 123 453 623 (질문의 예시)"""
    rsep = rng.choice([" ", " ", " ", "  ", ",", ", ", ";", "; ", "\n", " | "])
    b = _Builder()
    br = rng.random() < 0.15
    if br:
        b.sep("[")
    for i, row in enumerate(m):
        if i:
            b.sep(rsep)
        for j, v in enumerate(row):
            b.num(v, j == 0)
    if br:
        b.sep("]")
    return b.result()


def _fmt_flat_square(m, rng):
    """행 구분 없이 원소만 나열한 정사각 행렬: 1 2 3 4 → 2x2"""
    esep = rng.choice([" ", " ", ",", ", ", "  "])
    br = rng.random() < 0.25
    b = _Builder()
    if br:
        b.sep("[")
    # 행 경계가 눈에 보이지 않으므로 첫 원소만 R, 나머지는 전부 B.
    # n×n으로 접는 것은 모델이 아니라 디코더(모양 prior)가 결정합니다.
    for k, v in enumerate(x for row in m for x in row):
        if k:
            b.sep(esep)
        b.num(v, k == 0)
    if br:
        b.sep("]")
    return b.result()


def _fmt_concat_all(m, rng):
    """한 자리 숫자 정사각 행렬을 전부 붙여 쓴 형식: 1234 → 2x2"""
    b = _Builder()
    for k, v in enumerate(x for row in m for x in row):
        b.num(v, k == 0)  # flat 형식과 같은 이유로 첫 원소만 R
    return b.result()


def _noise(text, labels, rng):
    """앞뒤 공백, 끝에 붙은 구분자 등 사소한 잡음"""
    if rng.random() < 0.15:
        text, labels = " " * rng.randint(1, 2) + text, [O] * 0 + labels
        labels = [O] * (len(text) - len(labels)) + labels
    if rng.random() < 0.1:
        tail = rng.choice([" ", ";", ",", "\n", "."])
        text, labels = text + tail, labels + [O] * len(tail)
    return text, labels


def sample(rng=random):
    """(원본 문자열, 정답 행렬(문자열 2차원 리스트), 정규화 문자, 클래스 id, 라벨) 하나를 생성"""
    fmt = rng.choices(
        ["delimited", "concat_rows", "flat_square", "concat_all"],
        weights=[0.55, 0.22, 0.17, 0.06],
    )[0]
    if fmt == "delimited":
        rows, cols = rng.randint(1, 6), rng.randint(1, 6)
        if rng.random() < 0.5:
            cols = rows  # 정사각 행렬 비중을 높임 (det 용도)
        style = rng.choice(["digit", "small", "small", "big", "decimal", "fraction"])
    elif fmt == "concat_rows":
        rows, cols = rng.randint(2, 6), rng.randint(2, 6)
        if rng.random() < 0.7:
            cols = rows
        style = "digit"
    elif fmt == "flat_square":
        rows = cols = rng.randint(2, 6)
        style = rng.choice(["digit", "small", "small", "big", "decimal", "fraction"])
    else:
        rows = cols = rng.randint(2, 4)
        style = "digit"

    m = _sample_values(rows, cols, rng, style)
    text, labels = {
        "delimited": _fmt_delimited,
        "concat_rows": _fmt_concat_rows,
        "flat_square": _fmt_flat_square,
        "concat_all": _fmt_concat_all,
    }[fmt](m, rng)
    text, labels = _noise(text, labels, rng)
    chars, ids, labs = normalize(text, labels)
    return {"text": text, "matrix": m, "format": fmt, "chars": chars, "ids": ids, "labels": labs}


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(15):
        s = sample(rng)
        print(repr(s["text"]))
        print("   ", "".join("OIBR"[l] for l in s["labels"]), s["format"])
