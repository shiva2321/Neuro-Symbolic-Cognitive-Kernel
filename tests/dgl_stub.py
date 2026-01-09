"""Minimal DGL-like stub for tests that avoid native extensions."""
import torch
import networkx as nx
from typing import Iterable, List, Optional, Tuple, Union


TensorLike = Union[torch.Tensor, Iterable[int]]


def _to_int_list(values: TensorLike) -> List[int]:
    if torch.is_tensor(values):
        return [int(v) for v in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return [int(values)]


class DGLGraph:
    """Simplified graph object mimicking a subset of DGLGraph functionality."""

    def __init__(self, src: Optional[TensorLike] = None,
                 dst: Optional[TensorLike] = None,
                 num_nodes: Optional[int] = None):
        self._graph = nx.DiGraph()
        self.ndata = {}
        self.edata = {}

        if num_nodes is not None:
            self._graph.add_nodes_from(range(num_nodes))

        if src is not None and dst is not None:
            self.add_edges(src, dst)

    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        edges = list(self._graph.edges())
        src_list = [u for u, _ in edges]
        dst_list = [v for _, v in edges]
        return torch.tensor(src_list, dtype=torch.int64), torch.tensor(dst_list, dtype=torch.int64)

    def in_degrees(self) -> torch.Tensor:
        degrees = [self._graph.in_degree(n) for n in sorted(self._graph.nodes())]
        return torch.tensor(degrees, dtype=torch.int64)

    def out_degrees(self) -> torch.Tensor:
        degrees = [self._graph.out_degree(n) for n in sorted(self._graph.nodes())]
        return torch.tensor(degrees, dtype=torch.int64)

    def add_nodes(self, num_nodes: int):
        start = self.num_nodes()
        self._graph.add_nodes_from(range(start, start + num_nodes))

    def add_edges(self, src: TensorLike, dst: TensorLike):
        src_list = _to_int_list(src)
        dst_list = _to_int_list(dst)
        for u, v in zip(src_list, dst_list):
            self._graph.add_edge(int(u), int(v))

    def node_subgraph(self, nodes: TensorLike):
        node_list = _to_int_list(nodes)
        subgraph = self._graph.subgraph(node_list).copy()
        new_graph = DGLGraph()
        new_graph._graph = subgraph
        return new_graph

    def to_networkx(self) -> nx.DiGraph:
        return self._graph.copy()


def graph(data: Tuple[TensorLike, TensorLike], num_nodes: Optional[int] = None) -> DGLGraph:
    src, dst = data
    return DGLGraph(src=src, dst=dst, num_nodes=num_nodes)
