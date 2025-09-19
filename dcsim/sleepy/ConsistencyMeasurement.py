# dcsim/sleepy/ConsistencyMeasurement.py
from typing import Any, Dict, List, Tuple, cast, TYPE_CHECKING

from dcsim.framework import *
from .HonestNode import HonestNode

if TYPE_CHECKING:
    from .Configuration import Configuration


class ConsistencyMeasurement(MeasurementBase):
    """
    Minimal, zero-logging consistency checker with an "on-change" gate:
    it runs tests only when at least one node changed tip or length since
    the previous checked round.

    Modes (config.consistency_mode): "cp" | "reorg" | "both"
    Stride (config.consistency_stride):
        0  -> check every round
        >0 -> check only on multiples of `stride`
    """

    def __init__(
        self,
        honest_nodes: List["NodeBase"],
        adversary: "AdversaryControllerBase",
        trusted_third_parties: "TrustedThirdPartyBase",
        config: "Configuration",
    ) -> None:
        super().__init__(honest_nodes, adversary, trusted_third_parties, config)

        self._k: int = int(getattr(config, "confirm_time", 5))
        self._max_round: int = int(getattr(config, "max_round", 10**9))

        mode = str(getattr(config, "consistency_mode", "both")).lower()
        self._mode: str = mode if mode in ("cp", "reorg", "both") else "both"

        stride = int(getattr(config, "consistency_stride", 0))
        self._stride: int = stride if stride is not None else 0

        # Internal state
        self._stop: bool = False
        self._inconsistent: bool = False

        # Per-node cache of committed prefixes for "reorg > k" detection:
        # nid -> tuple(hash[:L_i])
        self._prev_committed: Dict[int, Tuple[str, ...]] = {}

        # Lightweight caches used by the on-change gate:
        # - last seen chain lengths
        # - last seen tip hashes
        self._prev_len: Dict[int, int] = {}
        self._prev_tip: Dict[int, str] = {}

    # -------- helpers --------
    @staticmethod
    def _chain(node: "NodeBase") -> List[Any]:
        """Return the node's main chain (as provided by HonestNode)."""
        return cast(HonestNode, node).main_chain

    @staticmethod
    def _prefix_tuple(chain: List[Any], upto: int) -> Tuple[str, ...]:
        """Return the tuple of block hashes up to (but including) index `upto-1`."""
        if upto <= 0:
            return ()
        upto = min(upto, len(chain))
        return tuple(b.hashval for b in chain[:upto])

    # -------- Runner API --------
    def should_stop(self, round: int) -> bool:
        """Stop when inconsistency is found or the maximum round is reached."""
        return self._stop or (round >= self._max_round)

    def report_round(self, round: int) -> None:
        """
        Called every round by the runner.
        Applies stride throttling, then the on-change gate; if a change is
        detected, runs the selected consistency checks.
        """
        if self._stop:
            return

        # Throttle via stride (if set)
        if self._stride > 0 and (round % self._stride != 0):
            if round >= self._max_round:
                self._stop = True
            return

        # 0) On-change gate: collect lengths and tips; see if anything changed
        changed = False
        lengths: Dict[int, int] = {}
        tips: Dict[int, str] = {}

        for n in self._honest_nodes:
            ch = self._chain(n)
            ln = len(ch)
            tip = ch[-1].hashval if ln > 0 else ""
            lengths[n.id] = ln
            tips[n.id] = tip
            if (self._prev_len.get(n.id, -1) != ln) or (self._prev_tip.get(n.id, "") != tip):
                changed = True

        # Update caches for next time (cheap O(#nodes))
        self._prev_len.update(lengths)
        self._prev_tip.update(tips)

        # If no node changed, outcomes cannot change either
        if not changed:
            if round >= self._max_round:
                self._stop = True
            return

        # 1) Build chains and eligible node set only when needed
        chains: Dict[int, List[Any]] = {}
        eligible_ids: List[int] = []
        for n in self._honest_nodes:
            ch = self._chain(n)
            chains[n.id] = ch
            if lengths[n.id] > self._k:
                eligible_ids.append(n.id)

        # 2) Common-Prefix(k)
        if self._mode in ("cp", "both"):
            if len(eligible_ids) >= 2:
                L = max(0, min(lengths[nid] for nid in eligible_ids) - self._k)
                if L > 0:
                    ref_id = eligible_ids[0]
                    ref_pref = self._prefix_tuple(chains[ref_id], L)
                    for nid in eligible_ids[1:]:
                        if self._prefix_tuple(chains[nid], L) != ref_pref:
                            self._inconsistent = True
                            self._stop = True
                            break
                    if self._stop:
                        return

        # 3) Reorg > k on individual nodes
        if self._mode in ("reorg", "both"):
            for n in self._honest_nodes:
                nid = n.id
                Li = max(0, lengths[nid] - self._k)
                cur_comm = self._prefix_tuple(chains[nid], Li)
                prev_comm = self._prev_committed.get(nid, ())
                pref_len = min(len(prev_comm), len(cur_comm))
                if prev_comm[:pref_len] != cur_comm[:pref_len]:
                    self._inconsistent = True
                    self._stop = True
                    break
                self._prev_committed[nid] = cur_comm

        # 4) Guard-rail on max_round
        if round >= self._max_round:
            self._stop = True

    def report_final(self) -> bool:
        """Return True if an inconsistency was detected during the run."""
        return self._inconsistent
