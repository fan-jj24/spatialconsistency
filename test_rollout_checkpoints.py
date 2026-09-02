import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import evaluate_rollout_checkpoints as evaluator
import run_gemini
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
        remote_model=legacy.GEMINI_MODEL,
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
    def test_remote_model_routes_maximum_thinking_level(self):
        self.assertEqual(
            rollout.remote_model_config("gemini-3.5-flash"),
            rollout.RemoteModelConfig("gemini", "high"),
        )
        self.assertEqual(
            rollout.remote_model_config("Qwen3.8-Flash-Next"),
            rollout.RemoteModelConfig("qwen_idealab", "xhigh"),
        )
        with self.assertRaisesRegex(ValueError, "仅支持"):
            rollout.remote_model_config("some-other-model")

    def test_remote_metadata_uses_selected_api_model(self):
        metadata = rollout.checkpoint_metadata(
            make_args(
                mode="remote", model_path=None,
                remote_model="Qwen3.8-Flash-Next",
            )
        )
        self.assertEqual(metadata["backend"], "qwen_idealab")
        self.assertEqual(metadata["remote_model"], "Qwen3.8-Flash-Next")
        self.assertEqual(metadata["remote_thinking_level"], "xhigh")

    def test_gemini_metadata_schema_and_resume_are_unchanged(self):
        args = make_args(mode="remote", name="gemini", model_path=None)
        row = make_row(7, 0)
        saved = {
            **rollout.checkpoint_metadata(args),
            "source_row": row.source_row,
            "ground_truth": row.ground_truth,
        }
        self.assertEqual(saved["gemini_thinking_level"], "high")
        self.assertNotIn("remote_thinking_level", saved)
        rollout.validate_checkpoint({row.source_row: saved}, [row], args)

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


class IdealabRequestTest(unittest.TestCase):
    def test_default_request_remains_gemini_high(self):
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
            text="",
        )
        with patch.object(run_gemini.requests, "post", return_value=response) as post:
            result = run_gemini.call_idealab("shared-key", "system", [])

        self.assertTrue(result["ok"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], run_gemini.MODEL)
        self.assertEqual(payload["thinking_config"], {"thinking_level": "high"})

    def test_selected_model_is_sent_to_idealab(self):
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
            text="",
        )
        with patch.object(run_gemini.requests, "post", return_value=response) as post:
            result = run_gemini.call_idealab(
                "shared-key", "system", [], model="Qwen3.8-Flash-Next",
                thinking_level="xhigh",
            )

        self.assertTrue(result["ok"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "Qwen3.8-Flash-Next")
        self.assertEqual(payload["thinking_config"], {"thinking_level": "xhigh"})


if __name__ == "__main__":
    unittest.main()
