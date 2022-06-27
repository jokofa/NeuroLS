#
from .ff_encoder import FFNodeEncoder
from .gnn_encoder import GNNNodeEncoder
from .sol_encoders import FlatSolEncoder, RNNSolEncoder

GRAPH_ENCODERS = [
    "GNNNodeEncoder"
]
