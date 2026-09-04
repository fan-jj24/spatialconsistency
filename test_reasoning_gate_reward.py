import unittest
from unittest import mock

import json_answer_reward as reward
import reward_model
import reward_model_client
import reward_model_server


QUESTION = """Are the two frames spatially consistent?
A. The spatial arrangement remains consistent.
B. The spatial arrangement has changed.
"""


class LastSentenceAndOptionTests(unittest.TestCase):
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

    def test_extracts_options_from_question(self):
        self.assertEqual(
            reward._extract_ab_options({"question": QUESTION}),
            (
                "The spatial arrangement remains consistent.",
                "The spatial arrangement has changed.",
            ),
        )

    def test_missing_options_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "A/B"):
            reward._extract_ab_options({"question": "No options here"})


class ReasoningGateCompositionTests(unittest.TestCase):
    def setUp(self):
        self.old_support = reward._reasoning_support_fn

    def tearDown(self):
        reward._reasoning_support_fn = self.old_support

    @staticmethod
    def _result(supported_option, unclear_probability=0.0):
        return {
            "supported_option": supported_option,
            "unclear_probability": unclear_probability,
        }

    def test_matching_supported_option_keeps_score(self):
        judge = mock.Mock(return_value=self._result("A"))
        reward._reasoning_support_fn = judge
        result = reward.compute_score_with_reasoning_gate(
            "spatial_consistency_pos",
            '<think>The layout is stable.</think>{"answer":"A"}',
            '{"answer":"A"}',
            {"question": QUESTION},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["R1"], 1.0)
        self.assertEqual(result["reasoning_gate"], 1.0)
        self.assertNotIn("acc", result)
        judge.assert_called_once_with(
            "The layout is stable.",
            "The spatial arrangement remains consistent.",
            "The spatial arrangement has changed.",
        )

    def test_opposite_supported_option_hard_zeros_score(self):
        reward._reasoning_support_fn = mock.Mock(return_value=self._result("B"))
        with mock.patch.object(
            reward,
            "compute_score",
            side_effect=AssertionError("downstream reward must be skipped"),
        ):
            result = reward.compute_score_with_reasoning_gate(
                "spatial_consistency_bbox_pos",
                '<think>The layout changed.</think>{"answer":"A","boxes":[]}',
                '{"answer":"A","summary":"same","boxes":[]}',
                {"question": QUESTION},
            )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["R1"], 1.0)
        self.assertEqual(result["reasoning_gate"], 0.0)

    def test_uses_question_options_instead_of_data_source_polarity(self):
        swapped_question = """Question
A. The arrangement changed.
B. The arrangement remained stable.
"""
        reward._reasoning_support_fn = mock.Mock(
            return_value=self._result("B")
        )
        result = reward.compute_score_with_reasoning_gate(
            "spatial_consistency_pos",
            '<think>The layout is stable.</think>{"answer":"B"}',
            '{"answer":"B"}',
            {"question": swapped_question},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reasoning_gate"], 1.0)

    def test_wrong_answer_skips_judge_and_downstream_rewards(self):
        judge = mock.Mock(side_effect=AssertionError("judge must be skipped"))
        reward._reasoning_support_fn = judge
        with mock.patch.object(
            reward,
            "compute_score",
            side_effect=AssertionError("downstream reward must be skipped"),
        ):
            result = reward.compute_score_with_reasoning_gate(
                "spatial_consistency_neg",
                '<think>The layout is stable.</think>{"answer":"A"}',
                '{"answer":"B"}',
                {"question": QUESTION},
            )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["R1"], 0.0)
        self.assertEqual(result["reasoning_gate_applied"], 0.0)
        judge.assert_not_called()

    def test_missing_thinking_hard_zeros_score(self):
        reward._reasoning_support_fn = mock.Mock(
            side_effect=AssertionError("judge must be skipped")
        )
        result = reward.compute_score_with_reasoning_gate(
            "spatial_consistency_pos",
            '{"answer":"A"}',
            '{"answer":"A"}',
            {"question": QUESTION},
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["reasoning_gate"], 0.0)
        self.assertEqual(result["reasoning_malformed"], 1.0)

    def test_unclear_does_not_penalize(self):
        reward._reasoning_support_fn = mock.Mock(
            return_value=self._result("U", unclear_probability=0.9)
        )
        result = reward.compute_score_with_reasoning_gate(
            "spatial_consistency_pos",
            '<think>More evidence is needed.</think>{"answer":"A"}',
            '{"answer":"A"}',
            {"question": QUESTION},
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reasoning_gate"], 1.0)

    def test_detection_has_no_answer_or_reasoning_gate(self):
        with mock.patch.object(reward, "compute_score", return_value=0.625):
            result = reward.compute_score_with_reasoning_gate(
                "spatial_detection", "{}", "{}", None
            )
        self.assertEqual(result["score"], 0.625)
        self.assertEqual(result["reasoning_gate"], 1.0)
        self.assertEqual(result["reasoning_gate_applied"], 0.0)


class RewardModelReasoningGateTests(unittest.TestCase):
    def test_argmax_returns_supported_question_option(self):
        model = reward_model.RewardModel()
        with (
            mock.patch.object(model, "_ensure_loaded"),
            mock.patch.object(model, "_truncate_conclusion", side_effect=lambda x: x),
            mock.patch.object(
                model,
                "_build_reasoning_gate_prompt",
                side_effect=lambda sentence, option_a, option_b: sentence,
            ),
            mock.patch.object(
                model,
                "_infer_label_probabilities",
                return_value=([[0.2, 0.7, 0.1]], [0.99]),
            ),
        ):
            result = model.classify_option_support("final", "same", "changed")

        self.assertEqual(result.supported_option, "B")
        self.assertEqual(result.unclear_probability, 0.1)


class ReasoningGateClientTests(unittest.TestCase):
    def test_sends_sentence_and_both_options(self):
        with mock.patch.object(
            reward_model_client,
            "_post_score",
            return_value={"supported_option": "A", "unclear_probability": 0.1},
        ) as post:
            result = reward_model_client.classify_option_support(
                "stable", "same", "changed"
            )
        self.assertEqual(result["supported_option"], "A")
        post.assert_called_once_with(
            "/classify-option-support",
            {"sentence": "stable", "option_a": "same", "option_b": "changed"},
            "reasoning gate",
        )


class ReasoningGateBatcherTests(unittest.TestCase):
    def test_scheduler_prefers_larger_ready_batch(self):
        scheduler = reward_model_server.QueueLengthInferenceScheduler()
        with scheduler._condition:
            scheduler._waiting["R4"] = (9, 2, lambda: 0)
            scheduler._waiting["reasoning_gate"] = (17, 1, lambda: 0)
            self.assertFalse(scheduler._may_start("R4"))
            self.assertTrue(scheduler._may_start("reasoning_gate"))

    def test_batcher_validates_structured_result(self):
        batcher = reward_model_server.DynamicBatcher(
            lambda items: [
                reward_model.OptionSupportScore(
                    supported_option="B",
                    unclear_probability=0.1,
                    probabilities=(0.2, 0.7, 0.1),
                    choice_mass=0.9,
                )
                for _ in items
            ],
            validate_result=reward_model_server._validate_reasoning_gate_result,
            task_name="reasoning_gate",
            max_batch_size=4,
            max_wait_ms=0,
            on_fatal=lambda exc: self.fail(str(exc)),
        )
        batcher.start()
        try:
            result = batcher.submit("final sentence", "same", "changed")
        finally:
            batcher.close()
        self.assertEqual(
            result,
            {"supported_option": "B", "unclear_probability": 0.1},
        )


if __name__ == "__main__":
    unittest.main()
