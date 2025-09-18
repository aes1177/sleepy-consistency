import hashlib
import pickle
from typing import *
from typing import Optional, List, NewType, cast
from dcsim.framework import *

"""
Lightweight transaction pool and block/tree utilities used by dcsim.

Key types:
- Tx        : transaction bytes
- Hashval   : 64-hex SHA-256 digest string
- Timestamp : integer round/time
"""

Tx = NewType("Tx", bytes)
Hashval = NewType("Hashval", str)
Timestamp = NewType("Timestamp", int)


class TxPool:
    """
    Minimal transaction pool backed by a set (deduplicated, O(1) membership).
    """

    def __init__(self) -> None:
        self.txs: Set[Tx] = set()

    def add_tx(self, tx: Tx) -> None:
        """Insert a transaction (no-op if already present)."""
        self.txs.add(tx)

    def remove_tx(self, tx: Tx) -> bool:
        """Remove a transaction; return True if it was present."""
        if tx in self.txs:
            self.txs.remove(tx)
            return True
        return False

    def find_tx(self, tx: Tx) -> Optional[Tx]:
        """Return tx if present, else None."""
        return tx if tx in self.txs else None

    def get_all(self) -> List[Tx]:
        """Return a new list copy to prevent external mutation."""
        return list(self.txs)

    def clear_all(self) -> None:
        """Drop all transactions."""
        self.txs.clear()


class TBlock:
    """
    Tree block wrapper with precomputed serialization and SHA-256 hash.

    Parameters
    ----------
    pbhv : Hashval
        Parent block hash value (string digest).
    txs : List[Tx]
        Transactions contained in the block.
    timestamp : Timestamp
        Logical time/round.
    pid : NodeId
        Producer node identifier.
    """

    def __init__(self, pbhv: Hashval, txs: List[Tx], timestamp: Timestamp, pid: NodeId) -> None:
        self.txs = txs
        self.pbhv = pbhv
        self.timestamp = int(timestamp)
        self.pid = pid
        self.children: List["TBlock"] = []

        # Precompute serialized payload and hash (stable across runs)
        payload = (self.pbhv, self.txs, self.timestamp, self.pid)
        self._serialized: bytes = pickle.dumps(payload)
        self._hashval: Hashval = cast(Hashval, hashlib.sha256(self._serialized).hexdigest())

    @property
    def id(self) -> int:
        """Producer numeric id."""
        return int(self.pid)

    @property
    def round(self) -> int:
        """Round/timestamp as int."""
        return int(self.timestamp)

    @property
    def hashval(self) -> Hashval:
        """Block hash (SHA-256 hex of serialized payload)."""
        return self._hashval

    @property
    def serialize(self) -> bytes:
        """Serialized payload used for hashing/networking."""
        return self._serialized

    @property
    def parent_hash(self) -> Hashval:
        """Alias for `pbhv` (read-only)."""
        return self.pbhv


class TNode:
    """
    Node of the block tree (points to a TBlock + parent/children bookkeeping).

    Invariants
    ----------
    - If a parent exists, `block.pbhv` must equal `father.hash`.
    - `index` holds a permutation of child indices; `index[0]` points to the
      preferred child (tip of the main chain under this node).
    """

    def __init__(self, depth: int, block: TBlock, father: Optional["TNode"]):
        self.depth: int = depth
        self.block: TBlock = block
        self.hash: Hashval = block.hashval
        self.father: Optional["TNode"] = father
        self.index: List[int] = []
        self.children: List["TNode"] = []
        self.num: int = 0

        if father:
            if block.pbhv != father.hash:
                raise ValueError("Parent hash mismatch: block.pbhv != father.hash")

    def get_children(self) -> List["TNode"]:
        """Return the children of this node."""
        return self.children

    def get_child_index(self, child_node: "TNode") -> int:
        """
        Return the *position in self.index* corresponding to child_node,
        or -1 if not found.
        """
        for i in range(len(self.index)):
            if child_node.hash == self.children[self.index[i]].hash:
                return i
        return -1

    def add_child(self, new_node: "TNode") -> bool:
        """
        Attach a child to this node (max fanout = 16). Returns True on success.
        If the node already has 16 children, the child is rejected.
        """
        if len(self.children) == 16:
            return False
        self.children.append(new_node)
        self.index.append(self.num)
        self.num += 1
        new_node.depth = self.depth + 1
        new_node.father = self
        return True

    def transfer_chain(self, i: int, j: int) -> None:
        """
        Swap the order of two children within the `index` permutation.
        """
        self.index[i], self.index[j] = self.index[j], self.index[i]

    def search(self, p_hash: Hashval) -> Optional["TNode"]:
        """
        Depth-first search in the subtree for a node with hash `p_hash`.
        Returns the node if found, else None.
        """
        res: Optional["TNode"] = None
        for child in self.children:
            if child.hash == p_hash:
                return child
            res = child.search(p_hash)
            if res is not None:
                break
        return res


# Genesis/super-root block (parent hash "0")
SuperRoot = TBlock(cast(Hashval, "0"), [], cast(Timestamp, 0), cast(NodeId, 0))


class BlockChain:
    """
    Simple tree-backed blockchain with a notion of a main chain (via `index[0]`).
    """

    def __init__(self) -> None:
        self.genesis = TNode(0, SuperRoot, None)
        self.head = self.genesis
        self.tail = self.genesis

    def find(self, hash_val: Hashval) -> Optional["TNode"]:
        """Find a node by block hash, starting from head."""
        if self.head.hash == hash_val:
            return self.head
        return self.head.search(hash_val)

    def add_child(self, t_node: "TNode", block: "TBlock") -> "TNode":
        """
        Add `block` as a child of `t_node`. If the new node extends a
        deeper path than the current tail, bubble it up along the path so
        that it becomes the preferred branch (index 0 at every ancestor).
        """
        new_node = TNode(t_node.depth + 1, block, t_node)
        t_node.add_child(new_node)

        if new_node.depth > self.tail.depth:
            temp_node = new_node
            while temp_node is not self.head:
                idx_in_perm = temp_node.father.get_child_index(temp_node)
                if idx_in_perm != 0 and idx_in_perm != -1:  # avoid `is` on ints
                    temp_node.father.transfer_chain(0, idx_in_perm)
                temp_node = temp_node.father
            self.tail = new_node
        return new_node

    @property
    def main_chain(self) -> List["TBlock"]:
        """
        Return the blocks along the main chain from genesis to tail,
        following `index[0]` at each step.
        """
        out: List[TBlock] = []
        temp_node = self.genesis
        while temp_node != self.tail:
            out.append(temp_node.block)
            i0 = temp_node.index[0]
            temp_node = temp_node.children[i0]
        out.append(self.tail.block)
        return out

    def get_top(self) -> TNode:
        """Return the current tail node (tip)."""
        return self.tail


class OrphanBlockPool:
    """
    Pool of blocks whose parent has not (yet) been connected to the tree.
    """

    def __init__(self) -> None:
        self.block: List[TBlock] = []

    def add_block(self, ablock: TBlock) -> None:
        """Insert an orphan block."""
        self.block.append(ablock)

    def pop_children(self, hv: Hashval) -> Optional[List[TBlock]]:
        """
        Remove and return all orphan blocks whose parent hash equals `hv`.
        Returns None if none were found.
        """
        matches = [b for b in self.block if b.pbhv == hv]
        if not matches:
            return None
        self.block = [b for b in self.block if b.pbhv != hv]
        return matches

    def find(self, hashval: Hashval) -> bool:
        """Check if an orphan with the given block hash exists."""
        return any(b.hashval == hashval for b in self.block)


def check_tx(tx: Tx) -> bool:
    """Basic tx well-formedness predicate (placeholder)."""
    return tx is not None


def _mix64(x: int) -> int:
    """
    SplitMix64-like mixer returning a non-negative 63-bit integer.
    Good enough for quick, reproducible pseudorandom thresholds.
    """
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31)
    return x & 0x7FFFFFFFFFFFFFFF  # keep 63 bits


def check_solution(tb: TBlock, probability: float) -> bool:
    """
    Lottery-like leader check:
    compute a deterministic 63-bit pseudorandom value from (pid, timestamp)
    and compare it against a threshold derived from `probability`.
    """
    pid = int(tb.pid)
    ts = int(tb.timestamp)
    z = _mix64((pid << 32) ^ ts)
    thr = int(probability * (1 << 63))
    return z < thr
