from typing import Any, List, Tuple, cast
from .MessageTuple import MessageTuple
from .NodeId import NodeId
from .NodeBase import NodeBase


class Context:
    """
    Per-round context passed to a node.

    If `compact_broadcast=True`, `broadcast(...)` does not expand into N messages.
    Instead it emits a single *logical* message (marked with `"__bcast__": True`)
    that the adversary will expand and deliver. This avoids O(N) message creation
    on the honest side and allows adversarial delivery control.
    """

    def __init__(
        self,
        nodes: Tuple[NodeId, ...],
        round: int,
        node: NodeBase,
        received_messages: Tuple[MessageTuple, ...],
        compact_broadcast: bool = False,
    ) -> None:
        self._nodes = nodes
        self._round = round
        self._node = node
        self._received_messages = received_messages
        self._message_tuples_to_send: List[MessageTuple] = []
        self._compact_broadcast = bool(compact_broadcast)

    @property
    def round(self) -> int:
        """Current round number (int)."""
        return self._round

    def send(self, receiver: NodeId, message: Any) -> None:
        """Queue a unicast message to `receiver`."""
        sender = self._node.id
        self._message_tuples_to_send.append(
            MessageTuple(sender=sender, receiver=receiver, round=self._round, message=message)
        )

    def broadcast(self, message: Any, include_self: bool = True) -> None:
        """
        Queue a broadcast message to all nodes.

        - If compact mode is OFF: expand to N (unicast) messages, optionally
          skipping the sender when `include_self=False`.
        - If compact mode is ON: enqueue a single logical message carrying
          the payload and `include_self` preference; the adversary expands it.
        """
        if self._compact_broadcast:
            # Single logical message (receiver=-1 acts as a sentinel)
            sender = self._node.id
            packed = {"__bcast__": True, "payload": message, "include_self": include_self}
            self._message_tuples_to_send.append(
                MessageTuple(sender=sender, receiver=cast(NodeId, -1), round=self._round, message=packed)
            )
            return

        for nid in self._nodes:
            if not include_self and nid == self._node.id:
                continue
            self.send(nid, message)

    @property
    def received_messages(self) -> Tuple[MessageTuple, ...]:
        """Messages delivered to this node at the start of the round."""
        return self._received_messages

    @property
    def messages_to_send(self) -> List[MessageTuple]:
        """Messages the node has queued to be sent this round."""
        return self._message_tuples_to_send
