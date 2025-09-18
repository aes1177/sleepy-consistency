from typing import *
import logging
from dcsim.framework import *
from .utils import *

if TYPE_CHECKING:
    from dcsim.sleepy.Configuration import Configuration
    # Optionally: from dcsim.framework import Context, TrustedThirdPartyCaller

# Message "type" tags used on the wire
TYPE_TX: Final[int] = 0
TYPE_BLOCK: Final[int] = 1


def _broadcast_excl_self(ctx: "Context", payload: dict) -> None:
    """
    Broadcast helper that excludes self when supported (older dcsim versions
    may not accept the 'include_self' kwarg).
    """
    try:
        ctx.broadcast(payload, include_self=False)
    except TypeError:
        ctx.broadcast(payload)


class HonestNode(NodeBase):
    """
    Honest node implementation (Sleepy-style):
    - Keeps a tx pool and an orphan block pool.
    - Maintains a block tree with a preferred/main chain.
    - Processes incoming txs/blocks, validates and gossips accepted blocks.
    - Mines a block (leader election) if the lottery check passes.

    Notes
    -----
    - Sleeping nodes (σ fraction) simply skip their round actions.
    - Authentication (FSign) and tx forwarding can be toggled via config flags.
    """

    def __init__(self, config: "Configuration") -> None:
        super().__init__(config)
        self._txpool = TxPool()
        self._orphanpool = OrphanBlockPool()
        self._block_chain = BlockChain()
        self._probability = config.probability
        self._sleeping = False  # support for sleeping nodes (sigma)

        # Performance / feature flags from config (default to False when missing)
        self._auth_checks = getattr(config, "auth_checks", False)
        self._enable_txs = getattr(config, "enable_txs", False)

    def set_sleeping(self, sleeping: bool) -> None:
        """Enable/disable sleeping mode (no processing, no mining)."""
        self._sleeping = bool(sleeping)

    def set_trusted_third_party(self, trusted_third_party: "TrustedThirdPartyCaller"):
        """Register FSign with the TTP facility."""
        super().set_trusted_third_party(trusted_third_party)
        self._trusted_third_party.call("FSign", "register")

    @property
    def main_chain(self) -> List[TBlock]:
        """Return the current main chain (genesis → tip)."""
        return self._block_chain.main_chain

    # ---- Orphan handling -----------------------------------------------------

    def recursive_remove_block_from_orphan_pool(self, block: TBlock) -> None:
        """
        Remove all orphan descendants of the given block (by parent hash match).
        """
        blocks_to_remove = self._orphanpool.pop_children(block.hashval)
        if blocks_to_remove is None:
            return
        for b2r in blocks_to_remove:
            self.recursive_remove_block_from_orphan_pool(b2r)

    def recursive_add_block_from_orphan_pool(self, curnode: TNode) -> None:
        """
        Attach all orphan children that reference curnode as parent, then recurse.
        """
        blocks_to_add = self._orphanpool.pop_children(curnode.block.hashval)
        if blocks_to_add is None:
            return
        for b2a in blocks_to_add:
            # Reject stale-timestamp children
            if curnode.block.timestamp >= b2a.timestamp:
                self.recursive_remove_block_from_orphan_pool(b2a)
            else:
                new_node = self._block_chain.add_child(curnode, b2a)
                self.recursive_add_block_from_orphan_pool(new_node)

    # ---- Per-round action ----------------------------------------------------

    def round_action(self, ctx: "Context") -> None:
        """
        Process inbox (txs/blocks), integrate valid blocks, and attempt to mine.

        The round number is retrieved robustly to avoid conflicts with Python's
        built-in `round()` (which might be returned if ctx.round isn't a property).
        """
        # Sleeping node: skip both processing and mining.
        if self._sleeping:
            return

        # Robustly extract the round number
        rnd_attr = getattr(ctx, "round", None)
        if callable(rnd_attr):  # in case it's not a @property
            try:
                rnd = int(rnd_attr())
            except TypeError:
                rnd = 0  # fallback: impossible to invoke built-in round() without args
        else:
            rnd = int(rnd_attr) if rnd_attr is not None else 0

        message_tuples = ctx.received_messages
        blocks: List[TBlock] = []  # accumulates validated, accepted blocks

        # 1) Process inbox -----------------------------------------------------
        for message_tuple in message_tuples:
            message = message_tuple.message
            sender: NodeId = message_tuple.sender

            # --- TRANSACTIONS ------------------------------------------------
            if message["type"] == TYPE_TX:
                if not self._enable_txs:
                    continue

                if self._auth_checks:
                    verified = self._trusted_third_party.call(
                        "FSign",
                        "verify",
                        signature=message["signature"],
                        message=message["value"],
                        sender_id=sender,
                    )
                else:
                    verified = True

                if verified and check_tx(message["value"]):
                    if not self._txpool.find_tx(message["value"]):
                        self._txpool.add_tx(message["value"])
                        my_sig = (
                            self._trusted_third_party.call("FSign", "sign", message=message["value"])
                            if self._auth_checks
                            else ""
                        )
                        _broadcast_excl_self(
                            ctx, {"type": TYPE_TX, "value": message["value"], "signature": my_sig}
                        )
                continue

            # --- BLOCKS -------------------------------------------------------
            elif message["type"] == TYPE_BLOCK:
                logging.debug(
                    "HonestNode.round_action: NodeId %s dealing with %s",
                    self.id,
                    message["value"],
                )

                # Obtain raw payload bytes (compat: property or method)
                ser = getattr(message["value"], "serialize", None)
                payload = ser if not callable(ser) else ser()

                if self._auth_checks:
                    verified = self._trusted_third_party.call(
                        "FSign", "verify", signature=message["signature"], message=payload, sender_id=sender
                    )
                else:
                    verified = True

                # Accept only valid leaders with non-future timestamps
                if verified and check_solution(message["value"], self._probability) and message["value"].timestamp <= rnd:
                    logging.debug(
                        "HonestNode.round_action: NodeId %s accepted message %s",
                        self.id,
                        message["value"].hashval,
                    )
                    blocks.append(message["value"])
                else:
                    logging.debug(
                        "HonestNode.round_action: NodeId %s declined message %s",
                        self.id,
                        message["value"].hashval,
                    )
                continue

        # 2) Integrate accepted blocks ----------------------------------------
        for block in blocks:
            # Skip duplicates (already known) or already-orphaned identical hash
            if self._block_chain.find(block.hashval) is not None:
                continue
            if self._orphanpool.find(block.hashval):
                continue

            logging.debug("HonestNode.round_action: NodeId %s received a new block", self.id)

            cur_node = self._block_chain.find(block.pbhv)
            if cur_node is None:
                # No parent yet → store as orphan
                logging.debug("HonestNode.round_action: NodeId %s received an orphan", self.id)
                self._orphanpool.add_block(block)

            elif cur_node.block.timestamp >= block.timestamp:
                # Reject non-increasing timestamps (also clear any descendants)
                self.recursive_remove_block_from_orphan_pool(block)
                logging.debug("HonestNode.round_action: NodeId %s received invalid time stamp", self.id)
                continue

            else:
                logging.debug("HonestNode.round_action: NodeId %s has added a block", self.id)
                # If extending the tip, evict txs included in the new block
                if cur_node == self._block_chain.get_top():
                    for tx in block.txs:
                        self._txpool.remove_tx(tx)

                new_node = self._block_chain.add_child(cur_node, block)
                # Try to attach any orphans now satisfied by this new node
                self.recursive_add_block_from_orphan_pool(new_node)

            # Gossip the (valid) block (even if it was an orphan)
            my_sig = (
                self._trusted_third_party.call(
                    "FSign",
                    "sign",
                    message=(block.serialize if not callable(getattr(block, "serialize", None)) else block.serialize()),
                )
                if self._auth_checks
                else ""
            )
            _broadcast_excl_self(ctx, {"type": TYPE_BLOCK, "value": block, "signature": my_sig})

        # 3) Mining (leader election) -----------------------------------------
        # Always use the robustly extracted round number `rnd`.
        pbhv = self._block_chain.get_top().block.hashval
        txs = self._txpool.get_all() if self._enable_txs else []
        my_block: TBlock = TBlock(pbhv, txs, cast(Timestamp, rnd), self.id)

        if check_solution(my_block, self._probability):
            logging.debug("HonestNode.round_action: NodeId %s chosen as the leader", self.id)
            self._block_chain.add_child(self._block_chain.get_top(), my_block)

            my_sig = (
                self._trusted_third_party.call(
                    "FSign",
                    "sign",
                    message=(
                        my_block.serialize
                        if not callable(getattr(my_block, "serialize", None))
                        else my_block.serialize()
                    ),
                )
                if self._auth_checks
                else ""
            )
            _broadcast_excl_self(ctx, {"type": TYPE_BLOCK, "value": my_block, "signature": my_sig})

            if self._enable_txs:
                self._txpool.clear_all()
