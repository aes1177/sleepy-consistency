# adversary_controller_base.py
import abc
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional

from .NodeBase import NodeBase

if TYPE_CHECKING:
    from .ConfigurationBase import ConfigurationBase
    from .NodeId import NodeId
    from .MessageTuple import MessageTuple
    from .TrustedThirdPartyCaller import TrustedThirdPartyCaller


class AdversaryControllerBase(metaclass=abc.ABCMeta):
    """
    Base class for the adversary controller.

    Responsibilities
    ----------------
    - Owns the set of corrupted node IDs and the list of honest node IDs.
    - Schedules deliveries, delays, and omissions of messages.
    - Maintains any adversarial state and exposes hooks the simulator calls
      each round.

    Notes
    -----
    - Concrete subclasses must implement:
        * get_delivered_messages(round)
        * round_action(round)
        * add_honest_node_messages(round, sender_id, messages_to_send)
    """

    def __init__(self, config: "ConfigurationBase") -> None:
        self._config = config
        # Generate corrupted node IDs deterministically via NodeBase
        self._corrupted_nodes: List["NodeId"] = [
            NodeBase.generate_node_id() for _ in range(config.num_corrupted_nodes)
        ]
        self._honest_nodes: Tuple["NodeId", ...] = tuple()
        self._trusted_third_parties: Dict["NodeId", "TrustedThirdPartyCaller"] = {}

    # --- Properties & helpers -------------------------------------------------

    @property
    def corrupted_node_list(self) -> Tuple["NodeId", ...]:
        """Immutable list of corrupted node IDs."""
        return tuple(self._corrupted_nodes)

    @property
    def honest_node_list(self) -> Tuple["NodeId", ...]:
        """Immutable list of honest node IDs (set by the simulator)."""
        return self._honest_nodes

    def is_corrupted(self, node_id: "NodeId") -> bool:
        """Return True iff node_id belongs to the corrupted set."""
        return node_id in self._corrupted_nodes

    def is_honest(self, node_id: "NodeId") -> bool:
        """Return True iff node_id belongs to the honest set."""
        return node_id in self._honest_nodes

    # --- Setup from simulator -------------------------------------------------

    def set_honest_node_list(self, node_ids: Tuple["NodeId", ...]) -> None:
        """
        Provide the list of honest node IDs.
        Also checks that honest/corrupted sets are disjoint (defensive).
        """
        if any(nid in self._corrupted_nodes for nid in node_ids):
            raise ValueError("ID collision between honest and corrupted nodes.")
        self._honest_nodes = node_ids

    def set_trusted_third_party(
        self, node_id: "NodeId", trusted_third_party: "TrustedThirdPartyCaller"
    ) -> None:
        """Register the TTP caller for a corrupted node."""
        self._trusted_third_parties[node_id] = trusted_third_party

    def get_trusted_third_party(self, node_id: "NodeId") -> Optional["TrustedThirdPartyCaller"]:
        """Return the TTP caller associated with a corrupted node, if any."""
        return self._trusted_third_parties.get(node_id)

    # --- Abstract API expected by the simulator -------------------------------

    @abc.abstractmethod
    def get_delivered_messages(self, round: int) -> List["MessageTuple"]:
        """
        Return the list of messages the adversary chooses to deliver in `round`.
        (Parameter name kept as `round` for compatibility across the codebase.)
        """

    @abc.abstractmethod
    def round_action(self, round: int) -> None:
        """
        Per-round adversary action (e.g., update internal state, schedule
        deliveries, apply delays/omissions).
        """

    @abc.abstractmethod
    def add_honest_node_messages(
        self, round: int, sender_id: "NodeId", messages_to_send: List["MessageTuple"]
    ) -> None:
        """
        Ingest messages produced by an honest node in `round` into the
        adversary-controlled buffer (to be delivered later).
        """
