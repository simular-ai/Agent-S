import logging
from typing import Any, Dict, List

logger = logging.getLogger("desktopenv.agent")


def agent_action(func):
    func.is_agent_action = True
    return func


class ACI:
    def __init__(self, top_app_only: bool = True, ocr: bool = False):
        self.top_app_only = top_app_only
        self.ocr = ocr
        self.index_out_of_range_flag = False
        self.notes: List[str] = []
        self.clipboard = ""
        self.nodes: List[Any] = []

    def get_active_apps(self, obs: Dict) -> List[str]:
        pass

    def get_top_app(self):
        pass

    def preserve_nodes(self, tree: Any, exclude_roles: set = None) -> List[Dict]:
        pass

    def linearize_and_annotate_tree(
        self, obs: Dict, show_all_elements: bool = False
    ) -> str:
        pass

    def find_element(self, element_id: int) -> Dict:
        pass
