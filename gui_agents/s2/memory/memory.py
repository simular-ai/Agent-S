import json
import os
from typing import Any, Dict, Optional


class Memory:
    def __init__(self, filepath: str = "memory.json"):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({}, f)

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def store(self, task_id: str, data: Dict[str, Any]) -> None:
        # Remove non-serializable fields (like raw screenshots)
        cleaned_data = {
            k: v for k, v in data.items() if not isinstance(v, bytes)
        }
        memory = self._load()
        memory[task_id] = cleaned_data
        self._save(memory)

    def retrieve(self, task_id: str) -> Optional[Dict[str, Any]]:
        memory = self._load()
        return memory.get(task_id)

    def update(self, task_id: str, new_data: Dict[str, Any]) -> None:
        memory = self._load()
        if task_id in memory:
            memory[task_id].update({
                k: v for k, v in new_data.items() if not isinstance(v, bytes)
            })
            self._save(memory)

    def clear(self, task_id: str) -> None:
        memory = self._load()
        if task_id in memory:
            del memory[task_id]
            self._save(memory)
