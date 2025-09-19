from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from .NodeId import NodeId
    from .TrustedThirdPartyBase import TrustedThirdPartyBase


class TrustedThirdPartyCaller:
    """
    Lightweight facade over a set of TTP instances.
    It binds a caller node_id and forwards calls to the named TTP,
    automatically injecting the caller id as the first argument.
    """

    def __init__(self, trusted_third_parties: List["TrustedThirdPartyBase"], node_id: "NodeId") -> None:
        # Map TTP name -> TTP instance
        self._trusted_third_parties: Dict[str, "TrustedThirdPartyBase"] = {t.name: t for t in trusted_third_parties}
        self._node: "NodeId" = node_id

    def call(self, party_name: str, function_name: str, *args, **kwargs) -> Any:
        """
        Forward a call to the named TTP:
            ttp.call(self._node, function_name, *args, **kwargs)

        Raises:
            ValueError: if the TTP name is unknown.
        """
        ttp = self._trusted_third_parties.get(party_name)
        if ttp is None:
            available = ", ".join(sorted(self._trusted_third_parties.keys())) or "<none>"
            raise ValueError(f"Unknown TTP '{party_name}'. Available: {available}")
        # IMPORTANT: unpack *args and **kwargs so they’re forwarded correctly.
        return ttp.call(self._node, function_name, *args, **kwargs)
