import torch
from torch import Tensor
from typing import Sequence, List, Dict, Any, Union
from collections.abc import Mapping
from data import GraphData
from numbers import Number


class GraphDataCollator:
    def __init__(self, config, follow_batch=[], exclude_keys=[]):
        self.config = config
        self.follow_batch = set(follow_batch)
        self.exclude_keys = set(exclude_keys)
        self.node_feature_key = config.get("node_feature_key", "x")
        self.edge_index_key = config.get("edge_index_key", "index")

    def __call__(self, batch: Sequence[GraphData]) -> GraphData:
        return self.collate_graph_data(batch)

    def collate_graph_data(self, batch: Sequence[GraphData]) -> GraphData:
        out = GraphData()

        for store_key in ["node", "edge", "graph"]:
            if store_key not in batch[0]:
                continue
            out_store = self.collate_store(batch, store_key)
            out[store_key] = out_store

        self.add_batch_info(out, batch)
        return out

    def collate_store(
        self, batch: Sequence[GraphData], store_key: str
    ) -> Dict[str, Any]:
        out_store = {}
        for attr in batch[0][store_key].keys():
            if attr in self.exclude_keys:
                continue

            values = [data[store_key][attr] for data in batch]
            out_store[attr] = self.collate_attribute(values)

            if attr in self.follow_batch:
                out_store[f"{attr}_batch"] = self.create_batch_vector(values)

        return out_store

    def add_batch_info(self, out: GraphData, batch: Sequence[GraphData]):
        if "node" in out and self.node_feature_key in out["node"]:
            out["node"]["batch"] = self.create_batch_vector([
                data["node"][self.node_feature_key] for data in batch
            ])
        if "edge" in out and self.edge_index_key in out["edge"]:
            out["edge"]["batch"] = self.create_edge_batch_vector(batch)

    def collate_attribute(self, values: List[Any]) -> Any:
        elem = values[0]
        elem_type = type(elem)
        if isinstance(elem, Tensor):
            return self.collate_tensor(values)
        elif isinstance(elem, float):
            return torch.tensor(values, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(values, dtype=torch.long)
        elif isinstance(elem, str):
            return values
        elif isinstance(elem, Mapping):
            return {
                key: self.collate_attribute([d[key] for d in values]) for key in elem
            }
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return elem_type(
                *(self.collate_attribute(samples) for samples in zip(*values))
            )
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate_attribute(samples) for samples in zip(*values)]
        else:
            raise TypeError(f"Cannot collate data of type {elem_type}")

    def collate_tensor(self, values: List[Tensor]) -> Tensor:
        if values[0].dim() == 0:
            return torch.stack(values, 0)

        # Check if all tensors have the same shape except for the first dimension
        shapes = [v.shape[1:] for v in values]
        if len(set(shapes)) > 1:
            return torch.cat([v.unsqueeze(0) for v in values], 0)

        return torch.cat(values, 0)

    def create_batch_vector(self, values: List[Tensor]) -> Tensor:
        batch = []
        for i, v in enumerate(values):
            batch.append(torch.full((v.size(0),), i, dtype=torch.long))
        return torch.cat(batch, dim=0)

    def create_edge_batch_vector(self, batch: Sequence[GraphData]) -> Tensor:
        edge_batches = []
        cumsum_nodes = 0
        for i, data in enumerate(batch):
            num_edges = data["edge"][self.edge_index_key].size(1)
            edge_batches.append(torch.full((num_edges,), i, dtype=torch.long))
            data["edge"][self.edge_index_key] += cumsum_nodes
            cumsum_nodes += data["node"][self.node_feature_key].size(0)
        return torch.cat(edge_batches, dim=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"node_feature_key={self.node_feature_key}, "
            f"edge_index_key={self.edge_index_key}, "
            f"follow_batch={list(self.follow_batch)}, "
            f"exclude_keys={list(self.exclude_keys)})"
        )


class PaddedGraphDataCollator(GraphDataCollator):
    def __init__(
        self,
        config,
        padding_config: Mapping[str, Mapping[str, Number]],
        follow_batch=[],
        exclude_keys=[],
    ):
        super().__init__(config, follow_batch, exclude_keys)
        self.padding_config = padding_config
        self.exclude_keys.update(
            key for store in padding_config.values() for key in store.keys()
        )

    def collate_store(
        self, batch: Sequence[GraphData], store_key: str
    ) -> Dict[str, Any]:
        out_store = super().collate_store(batch, store_key)

        if store_key in self.padding_config:
            for attr_key, padding_value in self.padding_config[store_key].items():
                if attr_key in out_store:
                    out_store[attr_key] = self.pad_attribute(
                        [data[store_key][attr_key] for data in batch], padding_value
                    )

        return out_store

    def pad_attribute(self, data_list: List[Tensor], padding_value: Number) -> Tensor:
        maxsize_data_to_pad, _ = torch.LongTensor([d.shape for d in data_list]).max(0)

        new_data_list = []
        for data in data_list:
            new_data = torch.full(
                size=tuple(maxsize_data_to_pad),
                fill_value=padding_value,
                device=data.device,
            )
            new_data[[slice(end) for end in data.shape]] = data
            new_data_list.append(new_data)

        return torch.stack(new_data_list, dim=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"config={self.config}, "
            f"padding_config={self.padding_config}, "
            f"follow_batch={list(self.follow_batch)}, "
            f"exclude_keys={list(self.exclude_keys)})"
        )
