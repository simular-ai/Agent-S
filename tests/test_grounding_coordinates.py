import io
import unittest
from unittest.mock import patch

from PIL import Image

from gui_agents.s3.agents.grounding import OSWorldACI
from gui_agents.s3.utils.common_utils import smart_resize


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _aci(screen_w, screen_h, grounding_w, grounding_h, **extra):
    engine = {"engine_type": "openai", "model": "gpt-5-2025-08-07", "api_key": "k"}
    grounding = {
        **engine,
        "grounding_width": grounding_w,
        "grounding_height": grounding_h,
        **extra,
    }
    return OSWorldACI(
        env=None,
        platform="linux",
        engine_params_for_generation=engine,
        engine_params_for_grounding=grounding,
        width=screen_w,
        height=screen_h,
    )


class TestSmartResize(unittest.TestCase):
    def test_matches_ui_tars_reference_for_1080p(self):
        # Reference values from the UI-TARS coordinate processing guide:
        # 1920x1080 is rounded to the nearest multiple of 28 in each dimension.
        self.assertEqual(smart_resize(1080, 1920), (1092, 1932))

    def test_dimensions_are_multiples_of_factor(self):
        for height, width in [(956, 1470), (1117, 1728), (1350, 2400), (600, 800)]:
            h_bar, w_bar = smart_resize(height, width)
            self.assertEqual(h_bar % 28, 0)
            self.assertEqual(w_bar % 28, 0)

    def test_downscales_images_above_max_pixels(self):
        h_bar, w_bar = smart_resize(3000, 6000)
        self.assertLessEqual(h_bar * w_bar, 16384 * 28 * 28)
        self.assertAlmostEqual(w_bar / h_bar, 2.0, delta=0.05)

    def test_upscales_images_below_min_pixels(self):
        h_bar, w_bar = smart_resize(10, 20)
        self.assertGreaterEqual(h_bar * w_bar, 100 * 28 * 28)


class TestResizeCoordinates(unittest.TestCase):
    def _ground(self, aci, screenshot_size, model_coords):
        """Simulate one grounding call that returned ``model_coords``."""
        obs = {"screenshot": _png(*screenshot_size)}
        with patch(
            "gui_agents.s3.agents.grounding.call_llm_safe",
            return_value=f"({model_coords[0]}, {model_coords[1]})",
        ):
            coords = aci.generate_coords("the thing", obs)
        return aci.resize_coordinates(coords)

    def test_1080p_screen_with_1080p_grounding_box_is_unchanged_at_center(self):
        aci = _aci(1920, 1080, 1920, 1080)
        resized_h, resized_w = smart_resize(1080, 1920)
        x, y = self._ground(aci, (1920, 1080), (resized_w // 2, resized_h // 2))
        self.assertEqual((x, y), (960, 540))

    def test_absolute_coordinates_follow_the_screenshot_actually_sent(self):
        # A 13" MacBook: logical screen 1470x956, screenshot sent at the same size.
        aci = _aci(1470, 956, 1920, 1080)
        resized_h, resized_w = smart_resize(956, 1470)
        # The model points at the bottom-right corner of the image it saw.
        x, y = self._ground(aci, (1470, 956), (resized_w, resized_h))
        self.assertEqual((x, y), (1470, 956))

    def test_absolute_coordinates_when_screenshot_is_downscaled(self):
        # A 1440p screen whose screenshot is capped at 2400px wide by the CLI.
        aci = _aci(2560, 1440, 1920, 1080)
        resized_h, resized_w = smart_resize(1350, 2400)
        x, y = self._ground(aci, (2400, 1350), (resized_w // 2, resized_h // 2))
        self.assertEqual((x, y), (1280, 720))

    def test_square_grounding_box_is_treated_as_normalized(self):
        # UI-TARS-72B style output in a fixed 0-1000 space, independent of the image.
        aci = _aci(2560, 1440, 1000, 1000)
        x, y = self._ground(aci, (2400, 1350), (500, 250))
        self.assertEqual((x, y), (1280, 360))

    def test_explicit_fixed_mode_overrides_auto_detection(self):
        aci = _aci(1470, 956, 1920, 1080, grounding_coordinate_space="fixed")
        x, y = self._ground(aci, (1470, 956), (960, 540))
        self.assertEqual((x, y), (735, 478))

    def test_explicit_image_mode_overrides_auto_detection(self):
        aci = _aci(1920, 1080, 1000, 1000, grounding_coordinate_space="image")
        resized_h, resized_w = smart_resize(1080, 1920)
        x, y = self._ground(aci, (1920, 1080), (resized_w, resized_h))
        self.assertEqual((x, y), (1920, 1080))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _aci(1920, 1080, 1920, 1080, grounding_coordinate_space="pixels")

    def test_falls_back_to_grounding_box_before_any_grounding_call(self):
        aci = _aci(1920, 1080, 1920, 1080)
        self.assertEqual(aci.resize_coordinates([960, 540]), [960, 540])


class TestScreenshotToScreen(unittest.TestCase):
    def test_ocr_coordinates_are_scaled_from_screenshot_to_screen(self):
        aci = _aci(2560, 1440, 1920, 1080)
        aci.assign_screenshot({"screenshot": _png(2400, 1350)})
        self.assertEqual(aci.screenshot_to_screen([1200, 675]), [1280, 720])

    def test_ocr_coordinates_unchanged_when_screenshot_matches_screen(self):
        aci = _aci(1920, 1080, 1920, 1080)
        aci.assign_screenshot({"screenshot": _png(1920, 1080)})
        self.assertEqual(aci.screenshot_to_screen([100, 200]), [100, 200])


if __name__ == "__main__":
    unittest.main()
