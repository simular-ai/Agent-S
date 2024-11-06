from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

class UIElementBase(ABC):
    @abstractmethod
    def __init__(self, ref: Any = None):
        self.ref = ref

    @staticmethod
    @abstractmethod
    def nodeFromTree(tree_data: Any) -> Optional['UIElementBase']:
        """Create a UIElement instance from the provided tree data"""
        pass

    @abstractmethod
    def getAttributeNames(self) -> List[str]:
        pass

    @abstractmethod
    def attribute(self, key: str) -> Any:
        pass

    @abstractmethod
    def children(self) -> List[Any]:
        pass

    @staticmethod
    @abstractmethod
    def systemWideElement() -> Optional['UIElementBase']:
        pass

    @abstractmethod
    def role(self) -> str:
        pass

    @abstractmethod
    def position(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def size(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def isValid(self) -> bool:
        pass

    @abstractmethod
    def parse(self) -> Dict[str, Any]:
        """Parse element into a standard dictionary format"""
        pass

    @staticmethod
    @abstractmethod
    def get_current_applications(obs: Dict) -> List[str]:
        pass
