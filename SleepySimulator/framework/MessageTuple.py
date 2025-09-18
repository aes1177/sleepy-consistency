# message_tuple.py
from typing import TYPE_CHECKING, NamedTuple, Any

if TYPE_CHECKING:
    from .NodeId import NodeId

class MessageTuple(NamedTuple):
    """
    A network message created at a given round.
    """
    sender: 'NodeId'
    receiver: 'NodeId'
    round: int
    message: Any
