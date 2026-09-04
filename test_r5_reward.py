import unittest
from unittest import mock

import json_answer_reward as reward
import reward_model
import reward_model_server


class LastSentenceTests(unittest.TestCase):
    def test_extracts_last_english_sentence(self):
        thinking = reward._extract_thinking(
            "<think>First observation. Therefore the layout is inconsistent."
            "</think>\n{\"answer\":\"B\"}"
        )
        self.assertEqual(
            reward._extract_last_sentence(thinking),
            "Therefore the layout is inconsistent.",
        )

    def test_extracts_chinese_sentence_without_space(self):
        self.assertEqual(
            reward._extract_last_sentence("先检查人物位置。因此画面空间一致。"),
            "因此画面空间一致。",
        )

    def test_accepts_opening_tag_in_prompt(self):
        self.assertEqual(
            reward._extract_thinking("Reasoning. Final conclusion.</think>{}"),
            "Reasoning. Final conclusion.",
        )


class R5CompositionTests(unittest.TestCase):
    def setUp(self):
        self.old_r5 = reward._r5_score_conclusion_fn

    def tearDown(self):
        reward._r5_score_conclusion_fn = self.old_r5

    @staticmethod
    def _mock_result(sentence, expected_stance):
        return {
            "score": 0.5,
            "conflict_probability": 1.0,
            "unclear_probability": 0.0,
        }

    def test_correct_answer_is_gated_and_acc_stays_binary(self):
        reward._r5_score_conclusion_fn = self._mock_result
        result = reward.compute_score_r5(
            "spatial_consistency_pos",
            '<think>The layout changed.</think>{"answer":"A"}',
            '{"answer":"A"}',
        )
        self.assertEqual(result["base_reward"], 1.0)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["r5_applied"], 1.0)

    def test_wrong_answer_skips_r5(self):
        r5 = mock.Mock(side_effect=AssertionError("R5 should be skipped"))
        reward._r5_score_conclusion_fn = r5
        result = reward.compute_score_r5(
            "spatial_consistency_neg",
            '<think>Consistent.</think>{"answer":"A"}',
            '{"answer":"B"}',
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["acc"], 0.0)
        self.assertEqual(result["r5_applied"], 0.0)
        r5.assert_not_called()

    def test_missing_thinking_gets_zero_gate(self):
        reward._r5_score_conclusion_fn = self._mock_result
        result = reward.compute_score_r5(
            "spatial_consistency_pos",
            '{"answer":"A"}',
            '{"answer":"A"}',
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["R5"], 0.0)
        self.assertEqual(result["reasoning_malformed"], 1.0)

    def test_non_r5_route_preserves_old_reward(self):
        result = reward.compute_score_r5(
            "vst_caption",
            '{"answer":"B"}',
            '{"answer":"B"}',
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["R5"], 1.0)
        self.assertEqual(result["r5_applied"], 0.0)


class RewardModelR5Tests(unittest.TestCase):
    def test_probability_gate_penalizes_only_opposite_stance(self):
        model = reward_model.RewardModel()
        with (
            mock.patch.object(model, "_ensure_loaded"),
            mock.patch.object(model, "_truncate_conclusion", side_effect=lambda x: x),
            mock.patch.object(model, "_build_r5_prompt", side_effect=lambda x: x),
            mock.patch.object(
                model,
                "_infer_label_probabilities",
                return_value=([[0.2, 0.7, 0.1]], [0.99]),
            ),
        ):
            consistent = model.score_conclusion("final", "consistent")
            inconsistent = model.score_conclusion("final", "inconsistent")

        expected_consistent_gate = 1.0 - (
            1.0 - reward_model.R5_CONFLICT_GATE
        ) * 0.7
        expected_inconsistent_gate = 1.0 - (
            1.0 - reward_model.R5_CONFLICT_GATE
        ) * 0.2
        self.assertAlmostEqual(consistent.gate, expected_consistent_gate)
        self.assertAlmostEqual(inconsistent.gate, expected_inconsistent_gate)
        self.assertEqual(consistent.unclear_probability, 0.1)


class R5BatcherTests(unittest.TestCase):
    def test_scheduler_prefers_larger_ready_batch(self):
        scheduler = reward_model_server.QueueLengthInferenceScheduler()
        with scheduler._condition:
            scheduler._waiting["R4"] = (9, 2, lambda: 0)
            scheduler._waiting["R5"] = (17, 1, lambda: 0)
            self.assertFalse(scheduler._may_start("R4"))
            self.assertTrue(scheduler._may_start("R5"))

    def test_scheduler_includes_requests_behind_ready_batch(self):
        scheduler = reward_model_server.QueueLengthInferenceScheduler()
        with scheduler._condition:
            scheduler._waiting["R4"] = (32, 1, lambda: 3)
            scheduler._waiting["R5"] = (32, 2, lambda: 20)
            self.assertFalse(scheduler._may_start("R4"))
            self.assertTrue(scheduler._may_start("R5"))

    def test_scheduler_uses_arrival_order_for_equal_batches(self):
        scheduler = reward_model_server.QueueLengthInferenceScheduler()
        with scheduler._condition:
            scheduler._waiting["R4"] = (12, 1, lambda: 0)
            scheduler._waiting["R5"] = (12, 2, lambda: 0)
            self.assertTrue(scheduler._may_start("R4"))
            self.assertFalse(scheduler._may_start("R5"))

    def test_r5_batcher_validates_structured_result(self):
        batcher = reward_model_server.DynamicBatcher(
            lambda pairs: [
                reward_model.ConclusionScore(
                    gate=0.75,
                    conflict_probability=0.5,
                    unclear_probability=0.1,
                    probabilities=(0.4, 0.5, 0.1),
                    choice_mass=0.9,
                )
                for _ in pairs
            ],
            validate_result=reward_model_server._validate_r5_result,
            task_name="R5",
            max_batch_size=4,
            max_wait_ms=0,
            on_fatal=lambda exc: self.fail(str(exc)),
        )
        batcher.start()
        try:
            result = batcher.submit("final sentence", "consistent")
        finally:
            batcher.close()
        self.assertEqual(
            result,
            {
                "score": 0.75,
                "conflict_probability": 0.5,
                "unclear_probability": 0.1,
            },
        )


if __name__ == "__main__":
    unittest.main()
