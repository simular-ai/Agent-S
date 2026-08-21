import unittest

from gui_agents.s3.utils.formatters import THOUGHTS_ANSWER_TAG_FORMATTER


class TestThoughtsAnswerFormatter(unittest.TestCase):
    def test_rejects_incomplete_tags(self):
        responses = [
            "plain response",
            "<thoughts>reasoning</thoughts>answer",
            "reasoning<answer>answer</answer>",
        ]

        for response in responses:
            with self.subTest(response=response):
                success, _ = THOUGHTS_ANSWER_TAG_FORMATTER(response)

                self.assertFalse(success)

    def test_accepts_complete_tags(self):
        success, _ = THOUGHTS_ANSWER_TAG_FORMATTER(
            "<thoughts>reasoning</thoughts><answer>answer</answer>"
        )

        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
