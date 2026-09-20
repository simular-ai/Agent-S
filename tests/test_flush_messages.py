import unittest
from types import SimpleNamespace

from gui_agents.s2_5.agents.worker import Worker as WorkerS2_5
from gui_agents.s3.agents.worker import Worker as WorkerS3


def _count_images(messages):
    return sum(
        1 for m in messages for c in m["content"] if "image" in c.get("type", "")
    )


class FlushMessagesMixin:
    worker_cls = None

    def _make_worker(self, messages, max_images=2, engine_type="openai"):
        # Build a Worker without running __init__ (avoids network / model setup).
        w = self.worker_cls.__new__(self.worker_cls)
        w.engine_params = {"engine_type": engine_type}
        w.max_trajectory_length = max_images
        w.generator_agent = SimpleNamespace(messages=messages)
        w.reflection_agent = SimpleNamespace(messages=[])
        return w

    def test_single_message_with_multiple_images(self):
        messages = [
            {"content": [{"type": "text"}] + [{"type": "image_url"}] * 4},
        ]
        self._make_worker(messages, max_images=2).flush_messages()
        self.assertEqual(_count_images(messages), 2)

    def test_images_across_messages(self):
        messages = [
            {"content": [{"type": "text"}, {"type": "image_url"}]} for _ in range(5)
        ]
        self._make_worker(messages, max_images=2).flush_messages()
        self.assertEqual(_count_images(messages), 2)

    def test_keeps_newest_images(self):
        messages = [
            {
                "content": [
                    {"type": "text"},
                    {"type": "image_url", "tag": "old"},
                    {"type": "image_url", "tag": "new"},
                ]
            }
        ]
        self._make_worker(messages, max_images=1).flush_messages()
        kept = [c for c in messages[0]["content"] if "image" in c.get("type", "")]
        self.assertEqual([c["tag"] for c in kept], ["new"])

    def test_text_entries_are_preserved(self):
        messages = [
            {"content": [{"type": "text"}] + [{"type": "image_url"}] * 3},
        ]
        self._make_worker(messages, max_images=0).flush_messages()
        self.assertEqual(messages[0]["content"], [{"type": "text"}])


class TestFlushMessagesS3(FlushMessagesMixin, unittest.TestCase):
    worker_cls = WorkerS3


class TestFlushMessagesS2_5(FlushMessagesMixin, unittest.TestCase):
    worker_cls = WorkerS2_5


if __name__ == "__main__":
    unittest.main()
