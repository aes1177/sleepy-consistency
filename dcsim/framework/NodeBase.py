# node_base.py
import abc
import sys
import random
from typing import Optional, Tuple, cast

from .NodeId import NodeId


class NodeBase(metaclass=abc.ABCMeta):
    """
    Abstract base class for all nodes (honest or adversarial-side helpers).

    Responsibilities
    ----------------
    - Holds a read-only node id (generated once).
    - Stores a reference to the simulation config and the full node list.
    - Provides a hook to inject a TrustedThirdPartyCaller.
    - Exposes the abstract per-round action `round_action(ctx)`.

    Notes
    -----
    - A class-level RNG is set by the Runner via `set_rng` to ensure a
      deterministic simulation; it is used for node id generation.
    - Node ids are unique within a run; collisions are avoided with a set.
    """

    __slots__ = ("_config", "_id", "_nodes", "_trusted_third_party")

    # Class-level RNG (set by Runner) + allocated-id set to avoid collisions
    _rng: Optional[random.Random] = None
    _allocated_ids: set[int] = set()

    @classmethod
    def set_rng(cls, rng: random.Random) -> None:
        """Install a shared RNG for deterministic behavior across the run."""
        cls._rng = rng

    def __init__(self, config) -> None:
        self._config = config
        self._id = self.generate_node_id()
        self._nodes: Tuple[NodeId, ...] = tuple()
        self._trusted_third_party = None  # set via `set_trusted_third_party`

    def set_trusted_third_party(self, trusted_third_party) -> None:
        """Attach a TrustedThirdPartyCaller (framework-provided facade)."""
        self._trusted_third_party = trusted_third_party

    @classmethod
    def generate_node_id(cls) -> "NodeId":
        """
        Generate a unique NodeId using the class RNG (or the global random module
        if none was set). Collisions are prevented with a small set.
        """
        rng = cls._rng or random
        # Rarely called compared to message processing; set membership is cheap.
        while True:
            candidate = rng.randint(1, sys.maxsize)
            if candidate not in cls._allocated_ids:
                cls._allocated_ids.add(candidate)
                return cast(NodeId, candidate)

    def set_node_list(self, node_ids: Tuple["NodeId", ...]) -> None:
        """Record the (immutable) list of all node ids in the simulation."""
        self._nodes = tuple(node_ids)

    @abc.abstractmethod
    def round_action(self, ctx) -> None:
        """Per-round behavior; must be implemented by subclasses."""
        ...

    @property
    def id(self) -> "NodeId":
        """Return this node's unique identifier."""
        return self._id
