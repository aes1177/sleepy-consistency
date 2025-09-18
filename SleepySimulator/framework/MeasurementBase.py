# measurement_base.py
import abc
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .AdversaryControllerBase import AdversaryControllerBase
    from .ConfigurationBase import ConfigurationBase
    from .NodeBase import NodeBase
    from .TrustedThirdPartyBase import TrustedThirdPartyBase

class MeasurementBase(metaclass=abc.ABCMeta):
    def __init__(
        self,
        honest_nodes: List['NodeBase'],
        adversary: 'AdversaryControllerBase',
        trusted_third_party: 'TrustedThirdPartyBase',
        config: 'ConfigurationBase'
    ) -> None:
        """
        Initialize the MeasurementBase.

        :param honest_nodes: honest nodes
        :param adversary: adversary controller in use
        :param trusted_third_party: shared TTP service (read-only or querying API)
        :param config: configuration in use
        """
        self._honest_nodes = honest_nodes
        self._adversary = adversary
        self._trusted_third_party = trusted_third_party
        self._config = config

    @abc.abstractmethod
    def should_stop(self, round: int) -> bool:
        """Return True to stop the simulation at the given round."""
        ...

    @abc.abstractmethod
    def report_final(self) -> None:
        """Report final conditions and outcome."""
        ...

    @abc.abstractmethod
    def report_round(self, round: int) -> None:
        """Report per-round node conditions."""
        ...
