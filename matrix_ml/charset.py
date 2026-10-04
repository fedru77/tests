"""문자 → 문자 클래스 변환과 입력 정규화.

모델은 원래 문자를 보지 않고 15개의 "문자 클래스"만 봅니다.
숫자 0~9는 모두 DIGIT 하나로 묶습니다. 원소를 어디서 끊을지는 숫자의 값이 아니라
배치(구분자, 그룹 길이, 괄호)로 정해지므로 이렇게 해도 정보 손실이 없고,
입력 차원이 줄어 모델이 작아집니다.
"""

# 문자 클래스 (모델 입력 one-hot의 인덱스)
CLASSES = [
    "DIGIT",      # 0-9
    "SPACE",      # 공백, 탭 (연속되면 하나로 압축)
    "NEWLINE",    # \n
    "COMMA",      # ,
    "SEMI",       # ;
    "LBRACK",     # [ ( {
    "RBRACK",     # ] ) }
    "MINUS",      # - − –
    "PLUS",       # +
    "DOT",        # .
    "SLASH",      # /
    "PIPE",       # |
    "AMP",        # &
    "BACKSLASH",  # \
    "OTHER",      # 문자, 기호 등 나머지 (연속되면 하나로 압축)
]
CLS = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

_CHAR_TO_CLASS = {
    " ": "SPACE", "\t": "SPACE", "\r": "SPACE", "　": "SPACE",
    "\n": "NEWLINE",
    ",": "COMMA", "，": "COMMA",
    ";": "SEMI",
    "[": "LBRACK", "(": "LBRACK", "{": "LBRACK",
    "]": "RBRACK", ")": "RBRACK", "}": "RBRACK",
    "-": "MINUS", "−": "MINUS", "–": "MINUS",
    "+": "PLUS",
    ".": "DOT",
    "/": "SLASH",
    "|": "PIPE",
    "&": "AMP",
    "\\": "BACKSLASH",
}

# 문자별 출력 라벨
LABELS = ["O", "I", "B", "R"]
O, I, B, R = 0, 1, 2, 3
# O: 숫자에 속하지 않음(구분자/잡음)
# I: 현재 원소(숫자)의 계속
# B: 새 원소 시작 (같은 행)
# R: 새 원소 시작 + 새 행 시작


def char_class(ch):
    if ch.isdigit() and ch.isascii():
        return CLS["DIGIT"]
    return CLS[_CHAR_TO_CLASS.get(ch, "OTHER")]


def normalize(text, labels=None):
    """입력 문자열을 (정규화된 문자 리스트, 클래스 id 리스트[, 라벨 리스트])로 바꿉니다.

    - 연속된 SPACE는 하나로, 연속된 OTHER(예: "begin{bmatrix}"의 글자들)도 하나로 압축합니다.
      의미는 그대로이고 시퀀스가 짧아져 RNN이 빨라집니다.
    - 앞뒤 공백은 제거합니다.
    - labels를 주면(학습 데이터 생성 시) 같은 방식으로 압축해서 함께 돌려줍니다.
    """
    chars, ids, labs = [], [], []
    for k, ch in enumerate(text):
        c = char_class(ch)
        if c == CLS["MINUS"]:
            ch = "-"  # 유니코드 마이너스(−, –)도 숫자로 읽을 수 있게 ASCII로
        if ids and c == ids[-1] and c in (CLS["SPACE"], CLS["OTHER"]):
            continue
        chars.append(ch)
        ids.append(c)
        if labels is not None:
            labs.append(labels[k])
    # 앞뒤 공백 제거
    while ids and ids[0] == CLS["SPACE"]:
        chars.pop(0); ids.pop(0)
        if labels is not None:
            labs.pop(0)
    while ids and ids[-1] == CLS["SPACE"]:
        chars.pop(); ids.pop()
        if labels is not None:
            labs.pop()
    if labels is not None:
        return chars, ids, labs
    return chars, ids
