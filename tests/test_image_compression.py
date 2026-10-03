import io
import unittest

from PIL import Image

from gui_agents.s3.utils.common_utils import compress_image_bytes


def _noisy_png(width: int, height: int) -> bytes:
    """Build a PNG that compresses poorly so it stays large, like a real desktop screenshot."""
    import random

    random.seed(0)
    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            (random.randrange(256), random.randrange(256), random.randrange(256))
            for _ in range(width * height)
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestCompressImageBytes(unittest.TestCase):
    def test_returns_input_unchanged_when_under_budget(self):
        data = _noisy_png(64, 64)
        self.assertIs(compress_image_bytes(data, max_bytes=len(data) + 1), data)

    def test_returns_input_unchanged_when_budget_is_none(self):
        data = _noisy_png(64, 64)
        self.assertIs(compress_image_bytes(data, max_bytes=None), data)

    def test_compresses_oversized_image_under_budget(self):
        data = _noisy_png(400, 300)
        budget = len(data) // 4
        result = compress_image_bytes(data, max_bytes=budget)
        self.assertLessEqual(len(result), budget)
        # Output must still be a decodable image.
        Image.open(io.BytesIO(result)).verify()

    def test_prefers_quality_reduction_before_downscaling(self):
        data = _noisy_png(400, 300)
        # Noisy RGB at JPEG quality 90 is far smaller than the PNG, so a
        # moderate budget should be met without shrinking the image.
        result = compress_image_bytes(data, max_bytes=len(data) // 2)
        self.assertEqual(Image.open(io.BytesIO(result)).size, (400, 300))

    def test_downscales_when_quality_reduction_is_not_enough(self):
        data = _noisy_png(400, 300)
        result = compress_image_bytes(data, max_bytes=len(data) // 40)
        self.assertLessEqual(len(result), len(data) // 40)
        width, height = Image.open(io.BytesIO(result)).size
        self.assertLess(width, 400)
        self.assertLess(height, 300)
        # Aspect ratio is preserved (within rounding).
        self.assertAlmostEqual(width / height, 400 / 300, places=1)

    def test_returns_smallest_attempt_when_budget_is_unreachable(self):
        data = _noisy_png(64, 64)
        result = compress_image_bytes(data, max_bytes=1)
        self.assertLess(len(result), len(data))
        Image.open(io.BytesIO(result)).verify()


if __name__ == "__main__":
    unittest.main()
