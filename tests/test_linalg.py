"""det / REF / RREF 계산 검증. python -m unittest discover -s tests -v"""
import itertools
import random
import unittest
from fractions import Fraction

from matrix_ml import linalg
from matrix_ml.solve import solve_text


def det_by_permutation(m):
    """정의대로 계산한 행렬식 (작은 행렬 검증용)"""
    n = len(m)
    total = Fraction(0)
    for perm in itertools.permutations(range(n)):
        inv = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
        prod = Fraction(1)
        for i in range(n):
            prod *= m[i][perm[i]]
        total += (-1) ** inv * prod
    return total


def is_ref(a, pivots):
    rows = len(a)
    for r, c in enumerate(pivots):
        if a[r][c] != 1 or any(a[r][j] != 0 for j in range(c)):
            return False
        if any(a[i][c] != 0 for i in range(r + 1, rows)):
            return False
    return all(all(x == 0 for x in a[r]) for r in range(len(pivots), rows))


class TestLinalg(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(linalg.det([[1, 2, 3], [4, 5, 3], [6, 2, 3]]), -45)
        self.assertEqual(linalg.det([[1, 2], [3, 4]]), -2)
        self.assertEqual(linalg.det([[0, 1], [1, 0]]), -1)  # 행 교환
        self.assertEqual(linalg.det([[1, 2], [2, 4]]), 0)
        self.assertEqual(linalg.det([[Fraction(1, 2), Fraction(1, 3)], [1, 1]]), Fraction(1, 6))
        rr, piv = linalg.rref([[1, 2, 3], [2, 4, 6]])
        self.assertEqual(rr, [[1, 2, 3], [0, 0, 0]])
        self.assertEqual(piv, [0])

    def test_det_non_square(self):
        with self.assertRaises(ValueError):
            linalg.det([[1, 2, 3]])

    def test_random_matrices(self):
        rng = random.Random(0)
        for _ in range(300):
            r, c = rng.randint(1, 5), rng.randint(1, 5)
            m = [[Fraction(rng.randint(-3, 3), rng.choice([1, 1, 2, 3])) for _ in range(c)] for _ in range(r)]
            if rng.random() < 0.3 and r > 1:
                m[-1] = [2 * x for x in m[0]]  # 일부러 rank를 떨어뜨림
            if r == c:
                self.assertEqual(linalg.det(m), det_by_permutation(m))
            a, piv = linalg.ref(m)
            self.assertTrue(is_ref(a, piv))
            b, piv2 = linalg.rref(m)
            self.assertEqual(piv, piv2)
            self.assertTrue(is_ref(b, piv2))
            for rr, cc in enumerate(piv2):  # 피벗 위쪽도 0
                self.assertTrue(all(b[i][cc] == 0 for i in range(rr)))
            if r == c:
                self.assertEqual(linalg.det(m) != 0, len(piv) == r)

    def test_steps_reproduce_result(self):
        steps = []
        linalg.rref([[0, 2, 4], [1, 1, 1], [2, 0, 1]], steps)
        self.assertEqual(steps[0], "R1 ↔ R2")


class TestSolveText(unittest.TestCase):
    def test_ambiguous_input_lists_all_candidates(self):
        out = solve_text("1 2 3 4")
        self.assertIn("가능한 해석 2개", out)
        self.assertIn("det = -2", out)
        self.assertIn("1x4 행렬은 정사각 행렬이 아니라", out)

    def test_clear_input_single_candidate(self):
        out = solve_text("123 453 623")
        self.assertNotIn("모호", out)
        self.assertIn("det = -45", out)


if __name__ == "__main__":
    unittest.main()
