import abc
from typing import TYPE_CHECKING, List, Callable, Dict, Any

if TYPE_CHECKING:
    from .NodeId import NodeId


class TrustedThirdPartyBase(metaclass=abc.ABCMeta):
    """
    Base class for TTP (Trusted Third Party) services.

    Subclasses must:
      - implement `round_action(self, round: int) -> None`
      - expose the callable API by returning bound methods from `_callable_functions`.
        Each exposed method must have the signature:
            method(caller: NodeId, *args, **kwargs) -> Any
    """

    def __init__(self, name: str) -> None:
        self._name = name
        # Index the callable methods once on construction.
        funcs = self._callable_functions
        self.__func: Dict[str, Callable[..., Any]] = {f.__name__: f for f in funcs}

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def round_action(self, round: int) -> None:
        """Per-round hook (optional for many TTPs)."""
        ...

    @property
    @abc.abstractmethod
    def _callable_functions(self) -> List[Callable[..., Any]]:
        """
        Return a list of bound methods (self.method) that the framework may invoke.
        Each method must accept: (caller: NodeId, *args, **kwargs).
        """
        ...

    def call(self, caller: "NodeId", function_name: str, *args, **kwargs) -> Any:
        """
        Dispatch a TTP function call with the given `caller` identity.

        Example
        -------
        ttp.call(caller_id, "sign", message=b"...")
        """
        func = self.__func.get(function_name)
        if func is None:
            available = ", ".join(sorted(self.__func.keys())) or "<none>"
            raise ValueError(f"TTP '{self._name}': unknown function '{function_name}'. Available: {available}")
        return func(caller, *args, **kwargs)
