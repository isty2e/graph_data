import copy
import torch
from torch import Tensor
from torch_geometric.data import (
    BaseData,
    FeatureStore,
    GraphStore,
    TensorAttr,
    EdgeAttr,
    EdgeLayout,
)
from torch_geometric.data.storage import BaseStorage
from torch_geometric.utils import is_sparse
from typing import Any, Optional, Union, Dict, List, Tuple
from storage import NodeDataStorage, EdgeDataStorage, GraphDataStorage


class GraphData(BaseData, FeatureStore, GraphStore):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__["_store"] = {
            "node": NodeDataStorage(_parent=self),
            "edge": EdgeDataStorage(_parent=self),
            "graph": GraphDataStorage(_parent=self),
        }
        for key, value in kwargs.items():
            self[key] = value

    def __getattr__(self, key: str) -> BaseStorage:
        if key in ("node", "edge", "graph"):
            return self._store[key]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'. The key should be one of 'node', 'edge', or 'graph'."
        )

    def __getitem__(self, key: str) -> Any:
        if key in self._store:
            return self._store[key]
        raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")

    def __setitem__(self, key: str, value: Any):
        if key not in self._store:
            raise KeyError(
                f"'{self.__class__.__name__}' object has no key '{key}'. Use 'node', 'edge', or 'graph'."
            )

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if key == "node":
                    self._check_node_dim(sub_key, sub_value)
                elif key == "edge":
                    self._check_edge_dim(sub_key, sub_value)
            self._store[key].update(value)
        else:
            raise TypeError(f"Value for '{key}' must be a dict, got {type(value)}")

    def __delitem__(self, key: str):
        if key in self._store:
            self._store[key].clear()
        else:
            raise KeyError(f"'{self.__class__.__name__}' object has no key '{key}'")

    def _check_node_dim(self, key: str, value: Any):
        if not isinstance(value, Tensor):
            raise TypeError(f"Node attribute '{key}' must be a Tensor")
        if self.num_nodes is not None and value.size(0) != self.num_nodes:
            raise ValueError(
                f"Node attribute '{key}' must have {self.num_nodes} entries in first dimension, got {value.size(0)}"
            )

    def _check_edge_dim(self, key: str, value: Any):
        if key == "index":
            if not isinstance(value, Tensor) or value.dim() != 2 or value.size(0) != 2:
                raise ValueError(
                    f"Edge index must be a Tensor of shape [2, num_edges], got {value.size()}"
                )
        else:
            if not isinstance(value, Tensor):
                raise TypeError(f"Edge attribute '{key}' must be a Tensor")
            if self.num_edges is not None and value.size(0) != self.num_edges:
                raise ValueError(
                    f"Edge attribute '{key}' must have {self.num_edges} entries in first dimension, got {value.size(0)}"
                )

    @property
    def num_nodes(self) -> Optional[int]:
        for value in self._store["node"].values():
            if isinstance(value, Tensor):
                return value.size(0)
        return None

    @property
    def num_edges(self) -> Optional[int]:
        if "index" in self._store["edge"]:
            return self._store["edge"]["index"].size(1)
        for value in self._store["edge"].values():
            if isinstance(value, Tensor):
                return value.size(0)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self._store["node"].to_dict(),
            "edge": self._store["edge"].to_dict(),
            "graph": self._store["graph"].to_dict(),
        }

    def to_namedtuple(self) -> Tuple:
        return (
            self._store["node"].to_namedtuple(),
            self._store["edge"].to_namedtuple(),
            self._store["graph"].to_namedtuple(),
        )

    def update(self, data: Union["GraphData", Dict[str, Any]]) -> "GraphData":
        for key, value in data.items():
            self[key] = value
        return self

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        def _recurse_cat_dim(key: str, value: Any):
            if is_sparse(value) and ("adj" in key or key == "index"):
                return (0, 1)
            elif key == "index":
                return -1
            elif isinstance(value, dict):
                for k, v in value.items():
                    result = _recurse_cat_dim(f"{key}.{k}", v)
                    if result != 0:
                        return result
            return 0

        for store_key in ["edge", "node", "graph"]:
            if key in self._store[store_key]:
                return _recurse_cat_dim(key, self._store[store_key][key])
        return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        def _recurse_inc(key: str, value: Any):
            if "batch" in key and isinstance(value, Tensor):
                return int(value.max()) + 1
            elif key == "index":
                return self.num_nodes
            elif isinstance(value, dict):
                for k, v in value.items():
                    result = _recurse_inc(f"{key}.{k}", v)
                    if result != 0:
                        return result
            return 0

        for store_key in ["edge", "node", "graph"]:
            if key in self._store[store_key]:
                return _recurse_inc(key, self._store[store_key][key])
        return 0

    def validate(self, raise_on_error: bool = True) -> bool:
        cls_name = self.__class__.__name__
        status = True

        try:
            if self.num_nodes is None:
                raise ValueError(f"'num_nodes' is undefined in '{cls_name}'")

            if self.num_edges is None:
                raise ValueError(f"'num_edges' is undefined in '{cls_name}'")

            for key, tensor in self._store["node"].items():
                if not isinstance(tensor, Tensor) or tensor.size(0) != self.num_nodes:
                    raise ValueError(
                        f"Node attribute '{key}' must be a Tensor with {self.num_nodes} entries in first dimension"
                    )

            for key, tensor in self._store["edge"].items():
                if key != "index" and (
                    not isinstance(tensor, Tensor) or tensor.size(0) != self.num_edges
                ):
                    raise ValueError(
                        f"Edge attribute '{key}' must be a Tensor with {self.num_edges} entries in first dimension"
                    )

            if "index" in self._store["edge"]:
                edge_index = self._store["edge"]["index"]
                if edge_index.dim() != 2 or edge_index.size(0) != 2:
                    raise ValueError(
                        f"'index' needs to be of shape [2, num_edges] in '{cls_name}' (found {edge_index.size()})"
                    )
                if edge_index.min() < 0:
                    raise ValueError(
                        f"'index' contains negative indices in '{cls_name}' (found {int(edge_index.min())})"
                    )
                if edge_index.max() >= self.num_nodes:
                    raise ValueError(
                        f"'index' contains larger indices than the number of nodes ({self.num_nodes}) in '{cls_name}' (found {int(edge_index.max())})"
                    )

        except ValueError as e:
            status = False
            if raise_on_error:
                raise e
            else:
                print(f"Validation Error: {str(e)}")

        return status

    # Implement required methods from FeatureStore and GraphStore interfaces
    def _put_tensor(self, tensor: Tensor, attr: TensorAttr) -> bool:
        out = self.get(attr.attr_name)
        if out is not None and attr.index is not None:
            out[attr.index] = tensor
        else:
            assert attr.index is None
            setattr(self, attr.attr_name, tensor)
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[Tensor]:
        tensor = getattr(self, attr.attr_name, None)
        if tensor is not None:
            return tensor[attr.index] if attr.index is not None else tensor
        return None

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        if hasattr(self, attr.attr_name):
            delattr(self, attr.attr_name)
            return True
        return False

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        return [
            TensorAttr(attr_name=name)
            for name in self._store.keys()
            if self._store["node"].is_node_attr(name)
            or self._store["edge"].is_edge_attr(name)
        ]

    def _put_edge_index(self, edge_index: Tensor, edge_attr: EdgeAttr) -> bool:
        self._store["edge"]["index"] = edge_index
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[Tuple[Tensor, Tensor]]:
        return self._store["edge"].get("index", None)

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        if "index" in self._store["edge"]:
            del self._store["edge"]["index"]
            return True
        return False

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        edge_attrs = []
        if "index" in self._store["edge"]:
            edge_attrs.append(EdgeAttr(layout=EdgeLayout.COO))
        return edge_attrs


# Example usage
data = GraphData(
    node={"x": torch.randn(10, 3)},
    edge={
        "index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "edge_attr": torch.randn(3, 2),
    },
    graph={"y": torch.tensor([1])},
)
print(data.num_nodes)  # Output: 10
print(data.num_edges)  # Output: 3
print(data.validate())  # Output: True
