import base64
from typing import Any, Callable, Dict, List, Optional
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature
from dcsim.framework import TrustedThirdPartyBase, NodeId


class FSignRSA(TrustedThirdPartyBase):
    """
    RSA-based signing/verification TTP (PSS padding, SHA-256).

    Intended for simulations where you want a PKI-like, non-trivial cost
    compared to hash/HMAC or the 'fast' signer. Keys are generated per node
    on first use (via register).

    Behavior
    --------
    - register(caller): generates a per-node RSA keypair if absent.
    - sign(caller, message): returns base64-encoded RSA-PSS signature (str).
    - verify(caller, signature, message, sender_id): verifies using sender's public key.

    Notes
    -----
    - Signatures are base64(text) for easy transport over JSON-like channels.
    - Messages can be bytes/bytearray/str (UTF-8) or any bytes()-convertible object.
    """

    def __init__(self, name: str, key_size: int = 2048, public_exponent: int = 65537) -> None:
        super().__init__(name)
        self._private_keys: Dict[NodeId, rsa.RSAPrivateKey] = {}
        self._key_size = int(key_size)
        self._public_exponent = int(public_exponent)

    @property
    def _callable_functions(self) -> List[Callable[..., Any]]:
        """Expose the TTP methods invokable via the framework."""
        return [self.register, self.sign, self.verify]

    def round_action(self, round: int) -> None:
        """Per-round hook (unused for this TTP)."""
        pass

    def register(self, caller: NodeId) -> None:
        """Ensure a private key exists for `caller`; generate if missing."""
        if caller not in self._private_keys:
            self._private_keys[caller] = rsa.generate_private_key(
                public_exponent=self._public_exponent,
                key_size=self._key_size,
                backend=default_backend(),  # kept for compatibility across cryptography versions
            )

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
            raise TypeError("FSignRSA.sign/verify: message must be bytes-like or str") from exc

    # --- signing / verification ---------------------------------------------

    def sign(self, caller: NodeId, message: Any) -> str:
        """
        Return a base64-encoded RSA-PSS(SHA-256) signature for `message`
        using the caller's private key.
        """
        if caller not in self._private_keys:
            self.register(caller)
        msg = self._ensure_bytes(message)
        signature = self._private_keys[caller].sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("ascii")

    def verify(self, caller: NodeId, signature: str, message: Any, sender_id: NodeId) -> bool:
        """
        Verify a base64-encoded RSA-PSS(SHA-256) `signature` for `message`
        using the public key of `sender_id`. Returns True/False.
        """
        try:
            pub = self._private_keys[sender_id].public_key()
        except KeyError:
            # Sender not registered → cannot verify
            return False

        msg = self._ensure_bytes(message)
        try:
            pub.verify(
                base64.b64decode(signature.encode("ascii")),
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False
        except Exception:
            # Any decoding/format errors should be treated as verification failure
            return False
