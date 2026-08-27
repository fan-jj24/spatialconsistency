import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import annotate_val_cases as annotation


class _FakeImage:
    size = (1000, 1000)

    def copy(self):
        return self


class _RecordingDraw:
    def __init__(self):
        self.texts = []
        self.calls = []

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def rectangle(self, *args, **kwargs):
        self._record("rectangle", *args, **kwargs)

    def text(self, _position, value, **kwargs):
        self.texts.append(value)
        self._record("text", _position, value, **kwargs)

    def line(self, *args, **kwargs):
        self._record("line", *args, **kwargs)

    def polygon(self, *args, **kwargs):
        self._record("polygon", *args, **kwargs)

    def ellipse(self, *args, **kwargs):
        self._record("ellipse", *args, **kwargs)

    def arc(self, *args, **kwargs):
        self._record("arc", *args, **kwargs)


def _render(gt_boxes, pred_boxes):
    draw = _RecordingDraw()
    pil = ModuleType("PIL")
    pil.ImageDraw = SimpleNamespace(Draw=lambda _image, _mode: draw)
    with patch.dict(sys.modules, {"PIL": pil}):
        annotation.annotate_second_image(_FakeImage(), gt_boxes, pred_boxes)
    return draw


class AnnotateSecondImageTest(unittest.TestCase):
    def test_valid_3d_arrow_hides_only_its_text_label(self):
        draw = _render(
            [([250, 250, 750, 750], "move (1, 0, -1)")],
            [([100, 100, 300, 300], "delete")],
        )

        self.assertEqual(draw.texts, ["P 1"])
        self.assertTrue(any(name == "arc" for name, _, _ in draw.calls))
        self.assertTrue(any(name == "ellipse" for name, _, _ in draw.calls))

    def test_2d_and_zero_3d_vectors_keep_text_labels(self):
        draw = _render(
            [([100, 100, 300, 300], "move (400, 200)")],
            [([500, 500, 700, 700], "move (0, 0, 0)")],
        )

        self.assertEqual(draw.texts, ["GT 1", "P 1"])

    def test_direction_parser_preserves_3d_depth(self):
        bbox = [100, 200, 300, 400]

        self.assertEqual(annotation._direction("move (1, 2, -3)", bbox), (1, 2, -3))
        self.assertEqual(annotation._direction("move (250, 350)", bbox), (50, 50))
        self.assertIsNone(annotation._direction("move (nan, 0, 1)", bbox))


class HtmlArrowLabelFilterTest(unittest.TestCase):
    def test_removes_only_valid_3d_arrow_labels(self):
        text = (
            'prefix {"boxes":['
            '{"bbox":[0,0,100,100],"label":"move right (1, 0, -1)"},'
            '{"bbox":[100,100,200,200],"label":"move (300, 150)"},'
            '{"bbox":[200,200,300,300],"label":"move (0, 0, 0)"},'
            '{"bbox":[300,300,400,400],"label":"delete"}'
            ']} suffix'
        )

        filtered = annotation._without_3d_arrow_labels(text)

        self.assertNotIn("move right (1, 0, -1)", filtered)
        self.assertIn('"bbox":[0,0,100,100]', filtered)
        self.assertIn("move (300, 150)", filtered)
        self.assertIn("move (0, 0, 0)", filtered)
        self.assertIn("delete", filtered)
        self.assertTrue(filtered.startswith("prefix "))
        self.assertTrue(filtered.endswith(" suffix"))

    def test_filters_only_the_box_object_used_for_annotation(self):
        text = (
            '{"boxes":[{"bbox":[0,0,1,1],"label":"old (1, 0, -1)"}]} '
            '{"boxes":[{"bbox":[0,0,1,1],"label":"new (0, 1, -1)"}]}'
        )

        filtered = annotation._without_3d_arrow_labels(text)

        self.assertIn("old (1, 0, -1)", filtered)
        self.assertNotIn("new (0, 1, -1)", filtered)


class HtmlAnnotationControlsTest(unittest.TestCase):
    def test_has_five_comparison_keys_and_three_metrics(self):
        case = annotation.Case(
            order=0,
            jsonl_line=1,
            source="dataset",
            source_path=Path("source.jsonl"),
            source_row=0,
            ground_truth='{"boxes":[]}',
            prediction='{"boxes":[]}',
            image_paths=[],
            gemini_prediction="answer",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            annotation.build_html(
                [case], Path("rollout.jsonl"), "1", output, sources=["dataset"]
            )
            document = output.read_text(encoding="utf-8")

        for label in (
            "预测正确",
            "预测错误",
            "Gemini 正确",
            "Gemini 错误",
            "预测比 Gemini 好",
        ):
            self.assertIn(label, document)
        for key in ("predictionAccuracy", "geminiAccuracy", "betterRate"):
            self.assertIn(key, document)
        self.assertIn("prediction_verdict", document)
        self.assertIn("gemini_verdict", document)
        self.assertIn("better_than_gemini", document)
        self.assertNotIn("不确定 <span class=\"kbd\">", document)


if __name__ == "__main__":
    unittest.main()
