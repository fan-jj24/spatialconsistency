import sys
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


if __name__ == "__main__":
    unittest.main()
