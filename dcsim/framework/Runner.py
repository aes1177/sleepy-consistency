from collections import defaultdict
from typing import DefaultDict, List, Tuple, TYPE_CHECKING
import random
import logging

from .ConfigurationBase import ConfigurationBase
from .Context import Context
from .MessageTuple import MessageTuple
from .NodeId import NodeId
from .TrustedThirdPartyCaller import TrustedThirdPartyCaller
from .NodeBase import NodeBase

if TYPE_CHECKING:
    # typing-only: avoids importing at runtime
    from .TrustedThirdPartyBase import TrustedThirdPartyBase


class Runner:
    """
    Simulation orchestrator.

    Responsibilities
    ----------------
    - Instantiate honest nodes (and mark a σ-fraction as sleeping).
    - Wire up Trusted Third Parties (TTPs) for both honest and corrupted nodes.
    - Delegate message delivery to the adversary controller each round.
    - Drive per-round actions for honest nodes, adversary, and TTPs.
    - Report consistency each round via the measurement component and stop on
      success/failure or round cap.

    Notes
    -----
    - A deterministic RNG is used across the whole run (seed from config).
    - Compact broadcast is read from config and stored for future options.
    """

    def __init__(self, config: ConfigurationBase) -> None:
        self._config = config
        self._ttps: List["TrustedThirdPartyBase"] = []

        # Reserved for future options (not used directly here)
        self._compact_broadcast = bool(getattr(self._config, "compact_broadcast", True))

        # Deterministic RNG for the whole run
        seed = getattr(config, "seed", None)
        self._rng = random.Random(seed) if seed is not None else random.Random()

        # Propagate RNG to NodeBase
        NodeBase.set_rng(self._rng)

    def add_trusted_third_party(self, ttp: "TrustedThirdPartyBase") -> None:
        """Register a TTP instance to be available to nodes and the adversary."""
        self._ttps.append(ttp)

    def init(self) -> None:
        """Build all simulation components (honest nodes, adversary, measurement)."""
        # --- honest nodes ---
        self._honest_nodes: List[NodeBase] = []
        for _ in range(self._config.num_honest_nodes):
            node = self._config.honest_node_type(self._config)
            ttp = TrustedThirdPartyCaller(self._ttps, node.id)
            node.set_trusted_third_party(ttp)
            self._honest_nodes.append(node)

        # --- sleeping (σ applied to honest nodes) ---
        sigma = float(getattr(self._config, "sigma", 0.0) or 0.0)
        sleep_target = min(int(round(sigma * len(self._honest_nodes))), len(self._honest_nodes))
        self._rng.shuffle(self._honest_nodes)  # deterministic due to shared RNG
        for n in self._honest_nodes[:sleep_target]:
            if hasattr(n, "set_sleeping"):
                n.set_sleeping(True)
        # Stable order (useful for logging/metrics)
        self._honest_nodes.sort(key=lambda n: n.id)

        # --- adversary ---
        self._adversary = self._config.adversary_controller_type(self._config)
        honest_node_ids: Tuple[NodeId, ...] = tuple(node.id for node in self._honest_nodes)
        corrupted_ids: Tuple[NodeId, ...] = tuple(self._adversary.corrupted_node_list)  # immutable view
        self._node_ids: Tuple[NodeId, ...] = tuple(list(honest_node_ids) + list(corrupted_ids))

        self._adversary.set_honest_node_list(honest_node_ids)
        for corrupted_node_id in corrupted_ids:
            ttp = TrustedThirdPartyCaller(self._ttps, corrupted_node_id)
            self._adversary.set_trusted_third_party(corrupted_node_id, ttp)

        # Optionally expose the node list to honest nodes
        for node in self._honest_nodes:
            if hasattr(node, "set_node_list"):
                node.set_node_list(self._node_ids)

        # --- measurement ---
        self._measure = self._config.measurement_type(
            self._honest_nodes, self._adversary, self._ttps, self._config
        )

    def run(self) -> bool:
        """
        Execute the simulation loop until the measurement asks to stop.

        Returns
        -------
        bool
            Final measurement outcome (e.g., True if inconsistency found).
        """
        rnd = 0
        while True:
            rnd += 1

            # Tick TTPs (if they expose a round_action hook)
            for t in self._ttps:
                ra = getattr(t, "round_action", None)
                if callable(ra):
                    ra(rnd)

            # Network: messages delivered this round by the adversary
            pending_messages = self._adversary.get_delivered_messages(rnd)
            receiver_messages: DefaultDict[NodeId, List[MessageTuple]] = defaultdict(list)
            for msg in pending_messages:
                receiver_messages[msg.receiver].append(msg)

            # Compact debugging
            num_blk = sum(1 for m in pending_messages if m.message.get("type") == 1)
            num_tx = sum(1 for m in pending_messages if m.message.get("type") == 0)
            logging.debug(
                "[NET] round=%d delivered=%d (blk=%d, tx=%d)", rnd, len(pending_messages), num_blk, num_tx
            )

            # Honest nodes: execute their round
            honest_ids = tuple(n.id for n in self._honest_nodes)
            for node in self._honest_nodes:
                # BUGFIX: pass rnd (int), NOT the built-in round()
                ctx = Context(honest_ids, rnd, node, tuple(receiver_messages[node.id]))
                node.round_action(ctx)
                # forward produced messages to the adversary
                self._adversary.add_honest_node_messages(rnd, node.id, ctx.messages_to_send)

            # Adversary: per-round action
            self._adversary.round_action(rnd)

            # Metrics for this round
            self._measure.report_round(rnd)

            # Stop condition
            if self._measure.should_stop(rnd):
                break

        return self._measure.report_final()
