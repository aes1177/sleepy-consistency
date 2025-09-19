import hashlib
import hmac
from typing import Any, Callable, Dict, Optional, List
from dcsim.framework import TrustedThirdPartyBase, NodeId


class FSignHash(TrustedThirdPartyBase):
    """
    Per-node HMAC-SHA256 signing/verification TTP.

    Behavior
    --------
    - Secret key for a node is derived deterministically from (seed, node_id)
      using BLAKE2b-256. If no seed is provided, seed defaults to 0
      (still deterministic).
    - sign(message) returns an HMAC-SHA256 hex string.
    - verify(signature, message, sender_id) checks in constant time.

    Notes
    -----
    - Exposes the callable set {register, sign, verify} to match the framework.
    - Messages can be bytes/bytearray/str (UTF-8) or any bytes()-convertible object.
    """

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        super().__init__(name)
        self._seed: int = 0 if seed is None else int(seed)
        self._secret_keys: Dict[NodeId, bytes] = {}

    @property
    def _callable_functions(self) -> List[Callable[..., Any]]:
        """Methods exposed to the framework through the TTP interface."""
        return [self.register, self.sign, self.verify]

    def round_action(self, round: int) -> None:
        """Per-round hook (unused by this TTP)."""
        pass

    def _derive_secret_key(self, node_id: NodeId) -> bytes:
        """Deterministically derive a 32-byte secret with BLAKE2b(seed || node_id)."""
        h = hashlib.blake2b(digest_size=32)
        h.update(str(self._seed).encode("utf-8"))
        h.update(str(int(node_id)).encode("utf-8"))
        return h.digest()

    def register(self, caller: NodeId) -> None:
        """Ensure a secret key exists for `caller`."""
        if caller not in self._secret_keys:
            self._secret_keys[caller] = self._derive_secret_key(caller)

    # --- input normalization -------------------------------------------------

    def _ensure_bytes(self, message: Any) -> bytes:
        """
        Convert supported inputs to bytes.

        Accepts: bytes, bytearray, str (UTF-8), or any object accepted by bytes(obj).
        """
        if isinstance(message, (bytes, bytearray)):
            return bytes(message)
        if isinstance(message, str):
            return message.encode("utf-8")
        try:
            return bytes(message)
        except Exception as exc:
            raise TypeError("FSignHash.sign/verify: message must be bytes-like or str") from exc

    # --- core HMAC logic -----------------------------------------------------

    def _sign(self, message: Any, sender_id: NodeId) -> str:
        """Compute HMAC-SHA256(message) using the sender's secret key."""
        if sender_id not in self._secret_keys:
            self.register(sender_id)
        msg = self._ensure_bytes(message)
        return hmac.new(self._secret_keys[sender_id], msg, hashlib.sha256).hexdigest()

    # --- TTP API -------------------------------------------------------------

    def sign(self, caller: NodeId, message: Any) -> str:
        """Return a hex-encoded HMAC signature for `message`."""
        return self._sign(message, caller)

    def verify(self, caller: NodeId, signature: str, message: Any, sender_id: NodeId) -> bool:
        """Verify `signature` against `message` using `sender_id`'s key (constant-time)."""
        expected = self._sign(message, sender_id)
        return hmac.compare_digest(expected, signature)
