import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import evaluate_rollout_checkpoints as evaluator
import rollout_checkpoint as rollout
import rollout_parquet_to_html as legacy


def make_args(**changes):
    values = dict(
        mode="local",
        name="qwen_a",
        internvl=False,
        data_path="/data/train.parquet",
        model_path="/models/qwen",
        num_samples=2,
        selection="first",
        seed=42,
        start_row=0,
        generation_seed=42,
        batch_size=1,
        dtype="bfloat16",
        max_prompt_length=2048,
        max_new_tokens=4096,
        temperature=0.01,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        reveal_gt_answer=False,
        gemini_thinking_level="high",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def make_row(source_row, order):
    return legacy.EvalRow(
        order=order,
        source_row=source_row,
        row={},
        ground_truth=json.dumps({"answer": "A"}),
        local_data_source=legacy.LOCAL_DATA_SOURCE,
        gemini_data_source=legacy.GEMINI_DATA_SOURCE,
    )


class CheckpointFormatTest(unittest.TestCase):
    def test_route_prefix_and_safe_name(self):
        self.assertEqual(
            rollout.checkpoint_filename("local", "qwen.step-10"),
            "local__qwen.step-10.jsonl",
        )
        with self.assertRaises(ValueError):
            rollout.checkpoint_filename("remote", "../escape")

    def test_retry_record_replaces_earlier_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local__qwen_a.jsonl"
            args = make_args(num_samples=1)
            row = make_row(7, 0)
            rollout.append_checkpoint(path, row, "old", "failed", args)
            rollout.append_checkpoint(path, row, "new", "", args)
            records = rollout.read_checkpoint(path)
        self.assertEqual(records[7]["prediction"], "new")
        self.assertEqual(records[7]["error"], "")

    def test_resume_rejects_a_different_sample_set(self):
        args = make_args(num_samples=1)
        saved_row = make_row(7, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local__qwen_a.jsonl"
            rollout.append_checkpoint(path, saved_row, "answer", "", args)
            completed = rollout.read_checkpoint(path)
        with self.assertRaisesRegex(ValueError, "不在本次抽样中"):
            rollout.validate_checkpoint(completed, [make_row(8, 0)], args)


class DynamicEvaluationTest(unittest.TestCase):
    def write_run(self, directory, args, predictions):
        path = Path(directory) / rollout.checkpoint_filename(args.mode, args.name)
        for order, (source_row, prediction) in enumerate(predictions):
            rollout.append_checkpoint(
                path, make_row(source_row, order), prediction, "", args
            )
        return evaluator.load_model_run(path)

    def test_prefix_controls_reward_route(self):
        with tempfile.TemporaryDirectory() as directory:
            local = self.write_run(
                directory, make_args(), [(10, "local-10"), (11, "local-11")]
            )
            remote = self.write_run(
                directory,
                make_args(
                    mode="remote", name="gemini", model_path=None,
                    gemini_thinking_level="high",
                ),
                [(10, "remote-10"), (11, "remote-11")],
            )
            with patch.object(
                evaluator.reward, "compute_score_details", return_value={"reward": 1}
            ) as local_score, patch.object(
                evaluator.reward, "score_answer_and_summary", return_value={"C": 1}
            ) as remote_score:
                scores, errors = evaluator.score_runs(
                    [local, remote], SimpleNamespace(reward_workers=1)
                )
        self.assertEqual(local_score.call_count, 2)
        self.assertEqual(remote_score.call_count, 2)
        self.assertFalse(any(errors.values()))
        self.assertEqual(scores[local.model_id][10], {"reward": 1.0})

    def test_html_contains_an_arbitrary_number_of_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [
                self.write_run(
                    root,
                    make_args(name=f"model_{index}"),
                    [(10, f"prediction {index}"), (11, "x")],
                )
                for index in range(4)
            ]
            scores = {run.model_id: {} for run in runs}
            errors = {run.model_id: {} for run in runs}
            output = root / "index.html"
            evaluator.build_html(
                output, Path("train.parquet"), [10, 11], runs, scores, errors,
                {}, {}, {},
            )
            document = output.read_text(encoding="utf-8")
        for index in range(4):
            self.assertIn(f"local:model_{index}", document)
        self.assertIn("本条最佳", document)


if __name__ == "__main__":
    unittest.main()
