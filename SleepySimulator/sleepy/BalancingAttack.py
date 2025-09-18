from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Any, cast
import hashlib
import logging
import random

from dcsim.framework import AdversaryControllerBase, MessageTuple, NodeId
from .utils import TBlock, SuperRoot, Timestamp, Tx

# Message type tags (match HonestNode)
TYPE_TX: int = 0
TYPE_BLOCK: int = 1


# --- PoW-like eligibility helper (aligned with honest nodes) ------------------
def _eligible(pid: int, rnd: int, probability: float) -> bool:
    """
    Deterministically decide if (pid, round) is eligible given per-round probability.
    """
    h = hashlib.sha256()
    h.update(int(pid).to_bytes(8, "big", signed=False))
    h.update(int(rnd).to_bytes(8, "big", signed=False))
    hv = int.from_bytes(h.digest(), "big")
    threshold = int((1 << 256) * float(probability))
    return hv < threshold


class BalancingAttack(AdversaryControllerBase):
    """
    Two-branch balancing adversary (A/B):

    - Replays honest blocks with fast delivery to the same side and delayed
      delivery (≤ Δ) to the opposite side; always respects the Δ bound.
    - Tracks and spends corrupted “leader slots” to keep branches balanced.
    - Supports compact broadcasts from honest nodes (no baseline re-broadcast).

    Strategy sketch
    ---------------
    1) Seed a fork when two honest siblings arrive within a Δ window:
       - deliver first sibling to side A (fast), set deadline = now + Δ
       - if a second sibling for the same parent arrives before the deadline,
         deliver it to side B (fast) and mark fork active.
    2) When one side grows, try to rebalance immediately using a corrupt slot
       (if eligible in the current round) or spend a banked past slot.
    3) All messages are eventually delivered to all nodes within Δ.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._probability: float = float(config.probability)
        self._delta: int = int(config.max_delay)

        # Network queues and per-receiver dedup
        self._pending_messages: DefaultDict[int, List[MessageTuple]] = defaultdict(list)
        self._seen_by_receiver: DefaultDict[NodeId, Set[Tuple[str, Any]]] = defaultdict(set)

        # RNG for stochastic delays
        self._rng = random.Random((getattr(config, "seed", 0) or 0) ^ 0xD15EA5)

        # Honest partitions A/B
        self._group_a: Tuple[NodeId, ...] = tuple()
        self._group_b: Tuple[NodeId, ...] = tuple()
        self._honest_set: Set[NodeId] = set()

        # Fork state
        self._fork_active: bool = False
        self._base_parent: Optional[str] = None
        self._tip_a: str = SuperRoot.hashval
        self._tip_b: str = SuperRoot.hashval

        # Balancing state
        self._last_grew: Optional[str] = None  # "A"/"B"/None
        self._deadline: Optional[int] = None   # Δ-window deadline while seeding/after growth

        # Banked corrupted slots: (timestamp, pid)
        self._banked_slots: List[Tuple[int, NodeId]] = []

        # Minimal tx pool for adversarial blocks
        self._txpool: Set[Tx] = set()

        # Stats (debug)
        self._offbranch_ignored_this_round = 0

    # --- wiring ---------------------------------------------------------------

    def set_honest_node_list(self, node_ids: Tuple[NodeId, ...]) -> None:
        super().set_honest_node_list(node_ids)
        self._honest_set = set(node_ids)
        half = len(node_ids) // 2
        self._group_a = tuple(node_ids[:half])
        self._group_b = tuple(node_ids[half:])
        logging.info("BalancingAttack: |A|=%d, |B|=%d", len(self._group_a), len(self._group_b))

    def set_trusted_third_party(self, node_id: NodeId, ttp) -> None:
        super().set_trusted_third_party(node_id, ttp)
        self._trusted_third_parties[node_id].call("FSign", "register")

    # --- dedup keys -----------------------------------------------------------

    def _msg_key(self, msg: Any) -> Tuple[str, Any]:
        """
        Create a receiver-local deduplication key from a compact-broadcast payload.
        """
        if msg["type"] == TYPE_TX:
            return ("tx", msg["value"])
        else:
            return ("blk", msg["value"].hashval)

    def _enqueue(self, delivery_round: int, t: MessageTuple, key_override: Optional[Tuple[str, Any]] = None) -> None:
        """
        Enqueue message for delivery at a future round, with per-receiver dedup.
        """
        seen = self._seen_by_receiver[t.receiver]
        key = key_override if key_override is not None else self._msg_key(t.message)
        if key in seen:
            return
        seen.add(key)
        self._pending_messages[delivery_round].append(t)

    # --- network --------------------------------------------------------------

    def get_delivered_messages(self, round: int) -> List[MessageTuple]:
        """Framework hook: release all messages scheduled for `round`."""
        return self._pending_messages.pop(round, [])

    def _sample_delay(self, prefer_fast: bool = False) -> int:
        """
        Sample an integer delay in [0, Δ]. In 'worst' mode:
          - prefer_fast=True  → 0
          - prefer_fast=False → Δ
        In 'stochastic' mode: Beta(a,b) scaled to [0, Δ].
        """
        Δ = self._config.max_delay
        mode = getattr(self._config, "delay_mode", "worst")
        if mode == "worst":
            return 0 if prefer_fast else Δ
        a = getattr(self._config, "delay_beta_a", 2.0)
        b = getattr(self._config, "delay_beta_b", 5.0)
        x = self._rng.betavariate(a, b)  # (0,1)
        d = int(round(x * Δ))
        return max(0, min(Δ, d))

    # --- delivery helpers -----------------------------------------------------

    def _deliver_honest_block_to_side(
        self, blk: TBlock, sig: str, orig_sender: NodeId, round: int, side: str
    ) -> None:
        """
        Replay an HONEST block with the original sender/signature (no adversary signing).
        Fast to the chosen side; delayed to the opposite side.
        """
        payload = {"type": TYPE_BLOCK, "value": blk, "signature": sig}
        same = self._sample_delay(prefer_fast=True)
        cross = self._sample_delay(prefer_fast=False)

        if side == "A":
            for nid in self._group_a:
                self._enqueue(round + same, MessageTuple(orig_sender, nid, round, payload))
            for nid in self._group_b:
                self._enqueue(round + cross, MessageTuple(orig_sender, nid, round, payload))
        else:
            for nid in self._group_b:
                self._enqueue(round + same, MessageTuple(orig_sender, nid, round, payload))
            for nid in self._group_a:
                self._enqueue(round + cross, MessageTuple(orig_sender, nid, round, payload))

    def _deliver_block_to_side(self, blk: TBlock, round: int, side: str) -> None:
        """
        Deliver an ADVERSARIAL block (signed by the adversary).
        If no corrupted nodes exist, do nothing.
        """
        if not self._corrupted_nodes:
            return
        sender = self._corrupted_nodes[0]
        ttp = self._trusted_third_parties[sender]
        sig = ttp.call("FSign", "sign", message=blk.serialize)
        payload = {"type": TYPE_BLOCK, "value": blk, "signature": sig}

        same = self._sample_delay(prefer_fast=True)
        cross = self._sample_delay(prefer_fast=False)

        if side == "A":
            for nid in self._group_a:
                self._enqueue(round + same, MessageTuple(sender, nid, round, payload))
            for nid in self._group_b:
                self._enqueue(round + cross, MessageTuple(sender, nid, round, payload))
        else:
            for nid in self._group_b:
                self._enqueue(round + same, MessageTuple(sender, nid, round, payload))
            for nid in self._group_a:
                self._enqueue(round + cross, MessageTuple(sender, nid, round, payload))

    def _mint_corrupt_block(self, parent_hash: str, ts: int, pid: NodeId) -> TBlock:
        """Create an adversarial block extending `parent_hash` at timestamp `ts`."""
        return TBlock(parent_hash, list(self._txpool), cast(Timestamp, ts), pid)

    # --- balancing ------------------------------------------------------------

    def _balance_with_current_corrupt_slot(self, round: int) -> None:
        """
        If a corrupted node is eligible in this round, mint immediately on the lagging side.
        """
        if not self._corrupted_nodes:
            return

        lag_side = "A" if self._last_grew == "B" else "B"
        parent = self._tip_a if lag_side == "A" else self._tip_b

        for pid in self._corrupted_nodes:
            if _eligible(int(pid), round, self._probability):
                blk = self._mint_corrupt_block(parent, round, pid)
                self._deliver_block_to_side(blk, round, lag_side)
                if lag_side == "A":
                    self._tip_a = blk.hashval
                else:
                    self._tip_b = blk.hashval
                self._last_grew = None
                self._deadline = None
                logging.debug("Balanced with current corrupt slot on side %s at round %d", lag_side, round)
                return

    def _spend_past_slot_if_any(self, round: int) -> bool:
        """
        Spend a previously banked corrupt slot (if any) to rebalance.
        Returns True if a slot was used.
        """
        if not self._banked_slots or not self._corrupted_nodes:
            return False

        lag_side = "A" if self._last_grew == "B" else "B"
        parent = self._tip_a if lag_side == "A" else self._tip_b

        ts, pid = self._banked_slots.pop()
        blk = self._mint_corrupt_block(parent, ts, pid)
        self._deliver_block_to_side(blk, round, lag_side)

        if lag_side == "A":
            self._tip_a = blk.hashval
        else:
            self._tip_b = blk.hashval

        self._last_grew = None
        self._deadline = None
        logging.debug(
            "Balanced by spending past slot ts=%d pid=%s on side %s at round %d", ts, pid, lag_side, round
        )
        return True

    # --- per-round events -----------------------------------------------------

    def round_action(self, round: int) -> None:
        """
        Per-round controller hook:
        - Bank a corrupted slot when no deadline is pending.
        - Otherwise try to rebalance using the current or a past slot.
        - Emit compact stats in debug logs.
        """
        # Bank a future slot if not currently racing a deadline
        if self._deadline is None:
            if self._corrupted_nodes:
                for pid in self._corrupted_nodes:
                    if _eligible(int(pid), round, self._probability):
                        self._banked_slots.append((round, pid))
                        logging.debug(
                            "Banked corrupt slot at round %d for pid=%s (s=%d)",
                            round,
                            pid,
                            len(self._banked_slots),
                        )
                        break
        else:
            # Try immediate rebalance with a current slot
            self._balance_with_current_corrupt_slot(round)
            # If deadline expires and no past slot is available → lose the race
            if self._deadline is not None and round >= self._deadline:
                ok = self._spend_past_slot_if_any(round)
                if not ok:
                    logging.info(
                        "BalancingAttack: deadline expired and no past slot left → losing at round %d", round
                    )

        if self._offbranch_ignored_this_round:
            logging.debug("[ADV] round=%d ignored_offbranch=%d", round, self._offbranch_ignored_this_round)
            self._offbranch_ignored_this_round = 0

        logging.debug(
            "[ADV] round=%d s=%d fork_active=%s last_grew=%s deadline=%s tipA=%s tipB=%s",
            round,
            len(self._banked_slots),
            self._fork_active,
            self._last_grew,
            self._deadline,
            self._tip_a[:8],
            self._tip_b[:8],
        )

    # --- handling honest messages --------------------------------------------

    def _try_start_or_extend_fork_with_honest_block(self, blk: TBlock, sig: str, round: int, sender: NodeId) -> None:
        """
        Seed a fork on first sibling; activate it if a second sibling arrives within Δ.
        While active, extend whichever side the honest block attaches to.
        """
        # Seeding phase
        if not self._fork_active:
            if self._base_parent is None:
                # First sibling: deliver to A, keep sibling window open for Δ rounds
                self._base_parent = blk.pbhv
                self._tip_a = blk.hashval
                self._tip_b = blk.pbhv
                self._deliver_honest_block_to_side(blk, sig, sender, round, "A")
                self._last_grew = "A"
                self._deadline = round + self._delta
                logging.debug("BalancingAttack: seed fork with A at round %d (deadline %d)", round, self._deadline)
                return
            else:
                # Second sibling within window → activate fork
                if blk.pbhv == self._base_parent and self._deadline is not None and round <= self._deadline:
                    self._tip_b = blk.hashval
                    self._deliver_honest_block_to_side(blk, sig, sender, round, "B")
                    self._fork_active = True
                    self._last_grew = None
                    self._deadline = None
                    logging.info("BalancingAttack: fork activated at round %d", round)
                    return
                # Still seeding: extend A if it attaches to tip A
                if blk.pbhv == self._tip_a:
                    self._tip_a = blk.hashval
                    self._deliver_honest_block_to_side(blk, sig, sender, round, "A")
                    self._last_grew = "A"
                    self._deadline = round + self._delta
                return

        # Active fork: extend the side that matches the parent
        if blk.pbhv == self._tip_a:
            self._tip_a = blk.hashval
            self._deliver_honest_block_to_side(blk, sig, sender, round, "A")
            if self._deadline is None:
                self._last_grew = "A"
                self._deadline = round + self._delta
        elif blk.pbhv == self._tip_b:
            self._tip_b = blk.hashval
            self._deliver_honest_block_to_side(blk, sig, sender, round, "B")
            if self._deadline is None:
                self._last_grew = "B"
                self._deadline = round + self._delta
        else:
            # Off-branch: not relevant to our managed fork
            self._offbranch_ignored_this_round += 1

    def _handle_new_messages(self, round: int, new_messages: List[MessageTuple]) -> None:
        """
        Consume compact-broadcast or normal messages emitted by honest nodes and
        feed the strategy (no direct enqueuing here).
        """
        for t in new_messages:
            msg = t.message
            if msg["type"] == TYPE_TX:
                # Tx pooling is optional (used when minting adversarial blocks)
                self._txpool.add(msg["value"])
            else:
                blk: TBlock = msg["value"]
                sig: str = msg.get("signature", "")
                # Assume honest validation already happened
                self._try_start_or_extend_fork_with_honest_block(blk, sig, round, t.sender)

    def add_honest_node_messages(
        self, round: int, sender_id: "NodeId", messages_to_send: List["MessageTuple"]
    ) -> None:
        """
        Framework hook: intercept messages emitted by honest nodes.
        Supports 'compact broadcast' (payload-only) without flooding the baseline network.
        """
        for t in messages_to_send:
            msg = t.message

            # --- compact broadcast from honest nodes ---
            if isinstance(msg, dict) and msg.get("__bcast__") is True:
                payload = msg["payload"]

                # (1) Update the attack logic ONLY (no expansion by baseline).
                if payload.get("type") == TYPE_BLOCK:
                    # Use original sender/signature
                    self._handle_new_messages(round, [MessageTuple(sender_id, sender_id, round, payload)])
                elif payload.get("type") == TYPE_TX:
                    self._txpool.add(payload["value"])

                # (2) Do not enqueue here; deliveries are done by *_deliver_* helpers.
                continue

            # --- normal messages (if any): for consistency, still route blocks here ---
            if isinstance(msg, dict) and msg.get("type") == TYPE_BLOCK:
                # Blocks no longer flow in the default channel, but handle defensively.
                self._handle_new_messages(round, [t])
                continue
