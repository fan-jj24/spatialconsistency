import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rollout_parquet_to_html as rollout


class PromptAnswerRevealTest(unittest.TestCase):
    def test_appends_natural_language_gt_answer_as_a_separate_message(self):
        row = {
            "prompt": [{"role": "user", "content": "describe the images"}],
            "reward_model": {
                "ground_truth": json.dumps({
                    "answer": "inconsistent",
                    "summary": "must not leak",
                    "boxes": [{"label": "must not leak either"}],
                })
            },
        }

        messages, images = rollout.build_messages_and_images(
            row, Path("dataset.parquet"), reveal_gt_answer=True
        )

        self.assertEqual(images, [])
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "describe the images"},
                {
                    "role": "user",
                    "content": (
                        "The correct answer is inconsistent. "
                        "Please complete the requested output based on this answer."
                    ),
                },
            ],
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("must not leak", serialized)
        self.assertNotIn('"answer"', serialized)

    def test_disabled_flag_keeps_original_prompt(self):
        row = {
            "prompt": [{"role": "user", "content": "original"}],
            "gts": {"answer": "inconsistent", "summary": "secret"},
        }

        messages, _ = rollout.build_messages_and_images(row, Path("data.parquet"))

        self.assertEqual(messages, row["prompt"])

    def test_reveal_requires_answer_field(self):
        row = {
            "prompt": [{"role": "user", "content": "original"}],
            "gt": {"summary": "no answer"},
        }

        with self.assertRaisesRegex(ValueError, "找不到 answer"):
            rollout.build_messages_and_images(
                row, Path("data.parquet"), reveal_gt_answer=True
            )


class GeminiConfigurationTest(unittest.TestCase):
    def test_loads_all_credentials_from_key_py(self):
        with tempfile.TemporaryDirectory() as directory:
            key_path = Path(directory) / "key.py"
            key_path.write_text(
                'ak="a"\nsk="s"\nep="e"\nbn="b"\napi_key="api"\n',
                encoding="utf-8",
            )

            credentials = rollout._load_gemini_credentials(key_path)

        self.assertEqual(
            credentials,
            rollout.GeminiCredentials("a", "s", "e", "b", "api"),
        )

    def test_reports_missing_key_py_field(self):
        with tempfile.TemporaryDirectory() as directory:
            key_path = Path(directory) / "key.py"
            key_path.write_text('ak="a"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sk, ep, bn, api_key"):
                rollout._load_gemini_credentials(key_path)

    def test_cli_defaults_and_new_flags(self):
        argv = [
            "rollout_parquet_to_html.py",
            "--model-path", "model",
            "--data-path", "data.parquet",
            "--out-dir", "output",
            "--reveal-gt-answer",
            "--reuse-gemini-dir", "previous",
            "--internvl-model-path", "internvl",
            "--internvl-batch-size", "2",
        ]
        with patch.object(sys, "argv", argv):
            args = rollout.parse_args()

        self.assertTrue(args.reveal_gt_answer)
        self.assertEqual(args.reuse_gemini_dir, "previous")
        self.assertEqual(args.internvl_model_path, "internvl")
        self.assertEqual(args.internvl_batch_size, 2)
        self.assertEqual(
            args.gemini_oss_prefix, "yk/ai-material/neo/fjj/rollout"
        )


class GeminiCheckpointReuseTest(unittest.TestCase):
    def test_copies_and_trusts_explicit_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "previous"
            current = root / "current"
            previous.mkdir()
            current.mkdir()
            checkpoint = {
                "source_row": 7,
                "data_path": "an-old-or-different-path.parquet",
                "ground_truth": "old metadata is explicitly trusted",
                "gemini_prediction": '{"answer":"inconsistent","summary":"copied"}',
                "gemini_error": "",
            }
            (previous / "gemini_results.jsonl").write_text(
                json.dumps(checkpoint) + "\n", encoding="utf-8"
            )

            destination = rollout._copy_reused_gemini_checkpoint(previous, current)
            row = rollout.EvalRow(
                order=0,
                source_row=7,
                row={},
                ground_truth="current metadata",
                local_data_source=rollout.LOCAL_DATA_SOURCE,
                gemini_data_source=rollout.GEMINI_DATA_SOURCE,
            )
            rollout.run_gemini_rollout(
                [row],
                Path("current.parquet"),
                current,
                destination,
                SimpleNamespace(),
                trusted_checkpoint=True,
            )

        self.assertEqual(row.gemini_prediction, checkpoint["gemini_prediction"])
        self.assertEqual(row.gemini_error, "")


class InternVLCheckpointTest(unittest.TestCase):
    def test_reuses_separate_internvl_checkpoint_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "internvl_results.jsonl"
            row = rollout.EvalRow(
                order=0,
                source_row=7,
                row={},
                ground_truth='{"answer":"A"}',
                local_data_source=rollout.LOCAL_DATA_SOURCE,
                gemini_data_source=rollout.GEMINI_DATA_SOURCE,
                internvl_prediction='{"answer":"A","summary":"internvl"}',
            )
            args = SimpleNamespace(
                internvl_model_path="internvl-model",
                internvl_batch_size=1,
                model_path="qwen-model",
                data_path="data.parquet",
                seed=42,
                generation_seed=42,
                batch_size=10,
                dtype="bfloat16",
                max_prompt_length=2048,
                temperature=0.01,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_new_tokens=4096,
                reveal_gt_answer=False,
                retry_errors=False,
            )
            internvl_args = SimpleNamespace(**vars(args))
            internvl_args.model_path = args.internvl_model_path
            internvl_args.batch_size = args.internvl_batch_size
            rollout._append_checkpoint(
                checkpoint_path,
                row,
                internvl_args,
                prediction_attr="internvl_prediction",
                error_attr="internvl_error",
            )
            row.internvl_prediction = ""

            with patch.object(
                rollout,
                "TransformersRollout",
                side_effect=AssertionError("checkpoint reuse must not load model"),
            ):
                rollout.run_internvl_rollout(
                    [row], Path("data.parquet"), checkpoint_path, args
                )

        self.assertEqual(
            row.internvl_prediction,
            '{"answer":"A","summary":"internvl"}',
        )
        self.assertEqual(row.internvl_error, "")


class InternVLRewardTest(unittest.TestCase):
    def test_scores_internvl_with_local_reward_route(self):
        row = rollout.EvalRow(
            order=0,
            source_row=1,
            row={},
            ground_truth="gt",
            local_data_source=rollout.LOCAL_DATA_SOURCE,
            gemini_data_source=rollout.GEMINI_DATA_SOURCE,
            internvl_prediction="internvl output",
        )
        args = SimpleNamespace(
            gemini=False,
            internvl_model_path="internvl-model",
            reward_workers=1,
        )
        with patch.object(
            rollout.reward,
            "compute_score_details",
            return_value={"reward": 0.75},
        ) as scorer:
            rollout.calculate_rewards([row], args)

        scorer.assert_called_once_with(
            rollout.LOCAL_DATA_SOURCE,
            "internvl output",
            "gt",
        )
        self.assertEqual(row.internvl_scores, {"reward": 0.75})


if __name__ == "__main__":
    unittest.main()
