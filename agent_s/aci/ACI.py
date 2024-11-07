from typing import Dict, List, Tuple, Any
import logging
import os

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

    def preserve_nodes(self, tree: Any, exclude_roles: set = None) -> List[Dict]:
        pass

    def linearize_and_annotate_tree(self, obs: Dict, show_all_elements: bool = False) -> str:
        screenshot = obs["screenshot"]
        
        # Handle different tree formats based on environment
        if self.is_osworld:
            # OSWorld XML string
            tree = self.UIElement.nodeFromTree(obs["accessibility_tree"])
        else:
            # Direct accessibility object
            tree = obs["accessibility_tree"]
            
        if not tree:
            logger.error("Failed to get accessibility tree")
            return "id\trole\ttitle\ttext"

        # TODO: write this function for each UIElement type 
        self.top_app = self.UIElement.get_top_app(obs)
        # self.top_app = UIElement.get_current_applications(obs: Dict)[0] if UIElement.get_current_applications(obs: Dict) else None
        
        preserved_nodes = self.preserve_nodes(tree)
        
        tree_elements = ["id\trole\ttitle\ttext"]
        for idx, node in enumerate(preserved_nodes):
            tree_elements.append(
                f"{idx}\t{node['role']}\t{node['title']}\t{node['text']}"
            )

        self.nodes = preserved_nodes
        return "\n".join(tree_elements)

    def find_element(self, element_id: int) -> Dict:
        try:
            return self.nodes[element_id]
        except IndexError:
            print("The index of the selected element was out of range.")
            self.index_out_of_range_flag = True
            return self.nodes[0]


