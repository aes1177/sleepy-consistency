# configuration_base.py
import abc
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .AdversaryControllerBase import AdversaryControllerBase
    from .MeasurementBase import MeasurementBase
    from .NodeBase import NodeBase


class ConfigurationBase(metaclass=abc.ABCMeta):
    """
    Abstract interface for wiring concrete types and top-level run parameters.

    Concrete configurations must specify:
      - the honest node class,
      - the adversary controller class,
      - the measurement/metrics class,
      - the number of honest and corrupted nodes.
    """

    @property
    @abc.abstractmethod
    def honest_node_type(self) -> Type["NodeBase"]:
        """Concrete honest node type."""
        ...

    @property
    @abc.abstractmethod
    def adversary_controller_type(self) -> Type["AdversaryControllerBase"]:
        """Concrete adversary controller type."""
        ...

    @property
    @abc.abstractmethod
    def measurement_type(self) -> Type["MeasurementBase"]:
        """Concrete measurement/metrics component type."""
        ...

    @property
    @abc.abstractmethod
    def num_honest_nodes(self) -> int:
        """Number of honest nodes."""
        ...

    @property
    @abc.abstractmethod
    def num_corrupted_nodes(self) -> int:
        """Number of corrupted (adversarial) nodes."""
        ...
