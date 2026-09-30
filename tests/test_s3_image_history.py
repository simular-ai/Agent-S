import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.agents.worker import Worker


def _message(label, with_image=True):
    content = [{"type": "text", "text": label}]
    if with_image:
        content.append({"type": "image_url", "image_url": {"url": label}})
    return {"role": "user", "content": content}


class TestS3ImageHistory(unittest.TestCase):
    def _worker(self, max_trajectory_length=8, max_history_images=None):
        with patch.object(Worker, "reset"):
            return Worker(
                worker_engine_params={"engine_type": "openai"},
                grounding_agent=object(),
                max_trajectory_length=max_trajectory_length,
                max_history_images=max_history_images,
            )

    def test_image_budget_defaults_to_trajectory_length(self):
        worker = self._worker(max_trajectory_length=5)

        self.assertEqual(worker.max_history_images, 5)

    def test_existing_positional_arguments_remain_compatible(self):
        with patch.object(Worker, "reset"):
            worker = Worker(
                {"engine_type": "openai"},
                object(),
                "ubuntu",
                5,
                False,
            )

        self.assertFalse(worker.enable_reflection)
        self.assertEqual(worker.max_history_images, 5)

    def test_explicit_image_budget_is_independent_of_trajectory_length(self):
        worker = self._worker(
            max_trajectory_length=8,
            max_history_images=3,
        )
        messages = [_message("system", with_image=False)] + [
            _message(f"turn-{index}") for index in range(5)
        ]
        worker.generator_agent = SimpleNamespace(messages=messages)
        worker.reflection_agent = None

        worker.flush_messages()

        image_labels = [
            part["image_url"]["url"]
            for message in messages
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        text_labels = [
            part["text"]
            for message in messages
            for part in message["content"]
            if part["type"] == "text"
        ]
        self.assertEqual(image_labels, ["turn-2", "turn-3", "turn-4"])
        self.assertEqual(
            text_labels,
            ["system", "turn-0", "turn-1", "turn-2", "turn-3", "turn-4"],
        )

    def test_agent_forwards_image_budget_to_worker(self):
        with patch("gui_agents.s3.agents.agent_s.Worker") as worker_class:
            AgentS3(
                {"engine_type": "openai"},
                object(),
                max_trajectory_length=8,
                max_history_images=3,
            )

        self.assertEqual(worker_class.call_args.kwargs["max_history_images"], 3)


if __name__ == "__main__":
    unittest.main()
