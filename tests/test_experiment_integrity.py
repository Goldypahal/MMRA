"""
test_experiment_integrity.py — Regression and integrity tests for MMRA framework.
"""

import unittest
import os
import asyncio
import pandas as pd
from src.graders import normalize, majority_vote, extract_final_answer, grade_exact, grade_numeric
from src.analysis import mcnemar_test, bootstrap_ci, debate_transitions
from src.client import APIResponse, is_mock_mode


class TestMMRAIntegrity(unittest.TestCase):

    def test_answer_normalization(self):
        self.assertEqual(normalize(" 42. "), "42")
        self.assertEqual(normalize("Sulfuric acid (H₂SO₄)."), "sulfuric acid h₂so₄")
        self.assertEqual(normalize("O(n)"), "on")

    def test_majority_vote_normalization(self):
        answers = ["42", "42.", " 42 \n", "43"]
        winner = majority_vote(answers)
        self.assertIn(winner, ["42", "42.", " 42 \n"])

    def test_extract_final_answer_markers(self):
        text_c4 = (
            "Reviewing others:\n"
            "Agent B is wrong.\n\n"
            "REVISED FINAL ANSWER: 15"
        )
        self.assertEqual(extract_final_answer(text_c4), "15")

        text_boxed = r"Step 1: calculate. \boxed{36}"
        self.assertEqual(extract_final_answer(text_boxed), "36")

    def test_mock_mode_strict_isolation(self):
        # Unless MMRA_MOCK_MODE is 1, is_mock_mode() MUST return False
        old_env = os.environ.get("MMRA_MOCK_MODE")
        try:
            os.environ["MMRA_MOCK_MODE"] = "0"
            self.assertFalse(is_mock_mode())
            os.environ["MMRA_MOCK_MODE"] = "1"
            self.assertTrue(is_mock_mode())
        finally:
            if old_env is not None:
                os.environ["MMRA_MOCK_MODE"] = old_env

    def test_mcnemar_stat(self):
        data = [
            {"task_id": 1, "condition": "C1", "score": 0.0},
            {"task_id": 1, "condition": "C4", "score": 1.0},
            {"task_id": 2, "condition": "C1", "score": 0.0},
            {"task_id": 2, "condition": "C4", "score": 1.0},
            {"task_id": 3, "condition": "C1", "score": 1.0},
            {"task_id": 3, "condition": "C4", "score": 1.0},
            {"task_id": 4, "condition": "C1", "score": 0.0},
            {"task_id": 4, "condition": "C4", "score": 0.0},
        ]
        df = pd.DataFrame(data)
        res = mcnemar_test(df, "C1", "C4")
        self.assertIsNotNone(res)
        self.assertEqual(res.a_wrong_b_correct, 2)
        self.assertEqual(res.a_correct_b_wrong, 0)

    def test_bootstrap_ci(self):
        data = [
            {"task_id": i, "condition": "C1", "score": 0.0} for i in range(10)
        ] + [
            {"task_id": i, "condition": "C4", "score": 1.0} for i in range(10)
        ]
        df = pd.DataFrame(data)
        ci = bootstrap_ci(df, "C1", "C4", n_boot=500)
        self.assertEqual(ci["mean_diff"], 1.0)
        self.assertEqual(ci["ci_lower"], 1.0)
    def test_omniroute_proxy_toggle(self):
        from src.config import MODELS
        from src.client import _make_client_for_model
        import src.config as config_mod

        old_val = getattr(config_mod, "USE_OMNIROUTE", False)
        try:
            config_mod.USE_OMNIROUTE = True
            client, model_name = _make_client_for_model(MODELS["A"])
            self.assertEqual(str(client.base_url).rstrip("/"), "http://localhost:20128/v1")

            config_mod.USE_OMNIROUTE = False
            client_direct, _ = _make_client_for_model(MODELS["A"])
            self.assertNotEqual(str(client_direct.base_url).rstrip("/"), "http://localhost:20128/v1")
        finally:
            config_mod.USE_OMNIROUTE = old_val


if __name__ == "__main__":
    unittest.main()
