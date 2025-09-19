# dcsim/sleepy/FSignFast.py
from typing import Callable, List, Optional
from dcsim.framework import TrustedThirdPartyBase, NodeId


class FSignFast(TrustedThirdPartyBase):
    """
    Ultra-light TTP signer/verifier used for fast simulations.

    Behavior:
    - `register(...)`: no-op (keeps no state).
    - `sign(...)`: returns an empty string (dummy signature).
    - `verify(...)`: always returns True.

    Notes
    -----
    - A `seed` parameter is accepted for interface compatibility with other TTPs
      (e.g., hash-based implementations) but is ignored here.
    - Method names and signatures match the expectations of the dcsim framework.
    """

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        super().__init__(name)
        # Kept for compatibility; not used by this fast implementation.
        self._seed = 0 if seed is None else int(seed)

    @property
    def _callable_functions(self) -> List[Callable]:
        """Expose the TTP methods invokable via the framework."""
        return [self.register, self.sign, self.verify]

    def round_action(self, round: int) -> None:
        """Per-round hook: no work needed for the fast signer."""
        pass

    def register(self, caller: NodeId) -> None:
        """Register a caller with the TTP (no state kept)."""
        pass

    def sign(self, caller: NodeId, message: bytes) -> str:
        """Return a dummy signature for the given message."""
        return ""

    def verify(self, caller: NodeId, signature: str, message: bytes, sender_id: NodeId) -> bool:
        """Trivially accept any signature/message pair."""
        return True
