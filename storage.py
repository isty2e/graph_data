from typing import Any, Optional, Dict
from torch import Tensor
from torch_geometric.data.storage import NodeStorage, EdgeStorage, BaseStorage


class NodeDataStorage(NodeStorage):
    @property
    def num_nodes(self) -> Optional[int]:
        for value in self._mapping.values():
            if isinstance(value, Tensor):
                return value.size(0)
            elif isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, Tensor):
                        return sub_value.size(0)
        return None

    def validate_dim(self) -> bool:
        expected_dim = self.num_nodes
        if expected_dim is None:
            return True  # 노드가 없는 경우도 유효하다고 간주
        return self._validate_dim_recursive(self._mapping, expected_dim)

    def _validate_dim_recursive(self, data: Dict[str, Any], expected_dim: int) -> bool:
        for value in data.values():
            if isinstance(value, Tensor):
                if value.size(0) != expected_dim:
                    return False
            elif isinstance(value, dict):
                if not self._validate_dim_recursive(value, expected_dim):
                    return False
        return True


class EdgeDataStorage(EdgeStorage):
    @property
    def num_edges(self) -> Optional[int]:
        for value in self._mapping.values():
            if isinstance(value, Tensor):
                return value.size(0)
            elif isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, Tensor):
                        return sub_value.size(0)
        return None

    def validate_dim(self) -> bool:
        expected_dim = self.num_edges
        if expected_dim is None:
            return True  # 엣지가 없는 경우도 유효하다고 간주
        return self._validate_dim_recursive(self._mapping, expected_dim)

    def _validate_dim_recursive(self, data: Dict[str, Any], expected_dim: int) -> bool:
        for value in data.values():
            if isinstance(value, Tensor):
                if value.size(0) != expected_dim:
                    return False
            elif isinstance(value, dict):
                if not self._validate_dim_recursive(value, expected_dim):
                    return False
        return True

    @property
    def edge_index(self) -> Optional[Tensor]:
        return self.get("index")

    @edge_index.setter
    def edge_index(self, edge_index: Optional[Tensor]) -> None:
        if edge_index is None:
            self.pop("index", None)
        else:
            self["index"] = edge_index


class GraphDataStorage(BaseStorage):
    @property
    def _key(self) -> str:
        key = self.__dict__.get("_key", None)
        if key is None or not isinstance(key, str):
            raise ValueError("'_key' does not denote a valid graph type")
        return key
