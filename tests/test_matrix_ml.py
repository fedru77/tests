"""python -m unittest discover -s tests -v"""
import os
import random
import unittest
from fractions import Fraction

import numpy as np

from matrix_ml import datagen, decode
from matrix_ml import model as M
from matrix_ml.charset import normalize
from matrix_ml.infer import DEFAULT_WEIGHTS, parse_matrix


class TestNormalize(unittest.TestCase):
    def test_collapses_spaces_and_letters(self):
        chars, ids = normalize("  \\begin{bmatrix} 1   2 \\end{bmatrix} ")
        self.assertEqual("".join(chars), "\\b{b} 1 2 \\e{b}")


class TestGradients(unittest.TestCase):
    def test_backprop_matches_numerical_gradient(self):
        rng = np.random.default_rng(0)
        p = M.init_params(4, seed=1)
        for k in p:
            p[k] += rng.normal(0, 0.3, p[k].shape)
        X = rng.integers(0, 15, (3, 6))
        Y = rng.integers(0, 4, (3, 6))
        L = [6, 3, 5]
        _, g = M.loss_and_grads(p, X, Y, L)
        for k in p:
            for idx in list(np.ndindex(p[k].shape))[::4]:
                old = p[k][idx]
                p[k][idx] = old + 1e-6
                lp, _ = M.loss_and_grads(p, X, Y, L)
                p[k][idx] = old - 1e-6
                lm, _ = M.loss_and_grads(p, X, Y, L)
                p[k][idx] = old
                num = (lp - lm) / 2e-6
                self.assertAlmostEqual(num, g[k][idx], delta=1e-5 + 1e-3 * abs(num), msg=f"{k}{idx}")


class TestDecoder(unittest.TestCase):
    def test_recovers_matrix_from_correct_labels(self):
        rng = random.Random(7)
        for _ in range(500):
            s = datagen.sample(rng)
            logp = np.full((len(s["ids"]), 4), np.log(0.1 / 3))
            logp[np.arange(len(s["ids"])), s["labels"]] = np.log(0.9)
            cands = decode.decode(logp, s["chars"], s["ids"])
            m = s["matrix"]
            if len(m) == 1 and decode._square_side(len(m[0])):
                # 1×n² 행은 prior 때문에 n×n이 1순위, 1×n²는 2순위여야 함
                self.assertEqual(cands[1]["matrix"], m, s["text"])
            else:
                self.assertEqual(cands[0]["matrix"], m, s["text"])

    def test_rows_always_equal_length(self):
        # 모델 출력이 엉망(균등 분포)이어도 디코더 결과는 항상 직사각형
        rng = random.Random(3)
        for _ in range(100):
            s = datagen.sample(rng)
            logp = np.log(np.full((len(s["ids"]), 4), 0.25))
            for c in decode.decode(logp, s["chars"], s["ids"]):
                self.assertEqual(len({len(row) for row in c["matrix"]}), 1)

    def test_square_option(self):
        chars, ids = normalize("1 2 3 4 5 6")
        logp = np.log(np.full((len(ids), 4), 0.25))
        for c in decode.decode(logp, chars, ids, square=True, top_k=10):
            self.assertEqual(c["shape"][0], c["shape"][1])


@unittest.skipUnless(os.path.exists(DEFAULT_WEIGHTS), "학습된 가중치 없음")
class TestTrainedModel(unittest.TestCase):
    CASES = [
        ("123 453 623", [[1, 2, 3], [4, 5, 3], [6, 2, 3]]),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
        ("1 2; 3 4", [[1, 2], [3, 4]]),
        ("1 -2 3\n4 5 -6\n7 8 9", [[1, -2, 3], [4, 5, -6], [7, 8, 9]]),
        ("12 34 56 78 90 11 12 13 14", [[12, 34, 56], [78, 90, 11], [12, 13, 14]]),
        ("1/2, 3/4; -5, 0.5", [[Fraction(1, 2), Fraction(3, 4)], [-5, Fraction(1, 2)]]),
        ("1 & 2 \\\\ 3 & 4", [[1, 2], [3, 4]]),
        ("[1 2 3] [4 5 6]", [[1, 2, 3], [4, 5, 6]]),
        ("1234", [[1, 2], [3, 4]]),
        ("−1 2; 3 −4", [[-1, 2], [3, -4]]),  # 유니코드 마이너스
    ]

    def test_examples(self):
        for text, want in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(parse_matrix(text)[0]["matrix"], want)


if __name__ == "__main__":
    unittest.main()
