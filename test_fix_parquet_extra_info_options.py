from pathlib import Path
import tempfile
import unittest

import fix_parquet_extra_info_options as fix

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


PROMPT = [
    {"role": "system", "content": "Answer the question."},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "left.jpg"},
            {
                "type": "text",
                "text": (
                    "Are the frames spatially consistent?\n"
                    "A. The spatial arrangement remains consistent.\n"
                    "B. The spatial arrangement has changed."
                ),
            },
        ],
    },
]


class ExtractOptionsTests(unittest.TestCase):
    def test_extracts_from_multimodal_messages(self):
        self.assertEqual(
            fix.extract_ab_options(PROMPT),
            (
                "The spatial arrangement remains consistent.",
                "The spatial arrangement has changed.",
            ),
        )

    def test_supports_parenthesized_and_option_prefix(self):
        prompt = "Question\n(A) yes\nOPTION B: no"
        self.assertEqual(fix.extract_ab_options(prompt), ("yes", "no"))

    def test_rejects_missing_option(self):
        with self.assertRaisesRegex(fix.DataValidationError, "缺少 B"):
            fix.extract_ab_options("Question\nA. yes")

    def test_rejects_ambiguous_options(self):
        with self.assertRaisesRegex(fix.DataValidationError, "多组"):
            fix.extract_ab_options("A. first\nB. no\nA. second")


class EnrichExtraInfoTests(unittest.TestCase):
    def test_adds_options_without_losing_existing_fields(self):
        enriched, status = fix.enrich_extra_info(
            "spatial_consistency_pos", PROMPT, {"index": 7, "split": "train"}
        )
        self.assertEqual(status, "added")
        self.assertEqual(enriched["index"], 7)
        self.assertEqual(enriched["split"], "train")
        self.assertEqual(
            enriched["option_a"], "The spatial arrangement remains consistent."
        )
        self.assertEqual(
            enriched["option_b"], "The spatial arrangement has changed."
        )

    def test_accepts_matching_existing_options(self):
        enriched, status = fix.enrich_extra_info(
            "spatial_consistency_neg",
            PROMPT,
            {
                "option_a": "The spatial arrangement remains consistent.",
                "option_b": "The spatial arrangement has changed.",
            },
        )
        self.assertEqual(status, "present")
        self.assertIsInstance(enriched, dict)

    def test_rejects_existing_options_that_disagree_with_prompt(self):
        with self.assertRaisesRegex(fix.DataValidationError, "不一致"):
            fix.enrich_extra_info(
                "spatial_consistency_bbox_neg",
                PROMPT,
                {"option_a": "changed", "option_b": "same"},
            )

    def test_detection_is_untouched(self):
        original = {"index": 3}
        enriched, status = fix.enrich_extra_info(
            "spatial_detection", [{"content": "no choices"}], original
        )
        self.assertEqual(status, "detection")
        self.assertEqual(enriched, original)

    def test_unknown_source_fails_closed(self):
        with self.assertRaisesRegex(fix.DataValidationError, "既不属于"):
            fix.enrich_extra_info("unexpected", PROMPT, {})


class DiscoveryTests(unittest.TestCase):
    def test_defaults_follow_rl_qwen_training_pool(self):
        self.assertEqual(
            fix.DEFAULT_DATA_ROOT, Path("/home/deepspeed/model_output/RL_qwen")
        )
        self.assertEqual(
            fix.DATASET_NAMES,
            (
                "consistent_qwen",
                "inconsistent_qwen",
                "detection_qwen",
                "consistent_all",
                "inconsistent_all",
            ),
        )

    def test_discovers_every_parquet_in_all_five_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for index, name in enumerate(fix.DATASET_NAMES):
                dataset_dir = root / name
                dataset_dir.mkdir()
                for filename in ("train_0000.parquet", "val_0000.parquet"):
                    path = dataset_dir / filename
                    path.touch()
                    expected.append(path)
                (dataset_dir / f"ignored_{index}.json").touch()

            files, missing = fix.discover_parquet_files(root)

            self.assertEqual(missing, [])
            self.assertEqual(files, sorted(expected))


@unittest.skipUnless(pa is not None and pq is not None, "pyarrow 未安装")
class ParquetRoundTripTests(unittest.TestCase):
    def test_adds_nested_fields_and_preserves_rows(self):
        rows = [
            {
                "data_source": "spatial_consistency_pos",
                "prompt": _prompt,
                "extra_info": {"index": index},
            }
            for index, _prompt in enumerate(
                (
                    "Question\nA. same\nB. changed",
                    "Question\nA. consistent\nB. inconsistent",
                )
            )
        ]
        rows.append(
            {
                "data_source": "spatial_detection",
                "prompt": "Detection prompt without choices",
                "extra_info": {"index": 2},
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train_0000.parquet"
            fixed = Path(directory) / "fixed.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=2)

            stats = fix.process_file(source, fixed)
            self.assertEqual(stats.rows, 3)
            self.assertEqual(stats.gated_rows, 2)
            self.assertEqual(stats.added_rows, 2)
            self.assertEqual(stats.detection_rows, 1)

            output = pq.read_table(fixed).to_pylist()
            self.assertEqual(output[0]["extra_info"]["index"], 0)
            self.assertEqual(output[0]["extra_info"]["option_a"], "same")
            self.assertEqual(output[0]["extra_info"]["option_b"], "changed")
            self.assertEqual(output[1]["extra_info"]["option_a"], "consistent")
            self.assertEqual(output[1]["extra_info"]["option_b"], "inconsistent")
            self.assertIsNone(output[2]["extra_info"]["option_a"])
            self.assertIsNone(output[2]["extra_info"]["option_b"])

            checked = fix.process_file(fixed)
            self.assertEqual(checked.added_rows, 0)
            self.assertEqual(checked.already_present_rows, 2)


if __name__ == "__main__":
    unittest.main()
