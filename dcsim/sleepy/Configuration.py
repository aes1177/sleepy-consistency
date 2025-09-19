from typing import Optional, Type
from dcsim.framework import ConfigurationBase
from .BalancingAttack import BalancingAttack
from .HonestNode import HonestNode
from .ConsistencyMeasurement import ConsistencyMeasurement


class Configuration(ConfigurationBase):
    """
    Simulation configuration container for the Sleepy model.

    Parameters
    ----------
    honest_node_type : Type[HonestNode]
        Concrete honest node class.
    adversary_controller_type : Type[BalancingAttack]
        Concrete adversary controller (e.g., BalancingAttack or ForkSustainAttack).
    measurement_type : Type[ConsistencyMeasurement]
        Concrete measurement class (consistency checker).
    num_honest_nodes : int
        Number of honest nodes.
    num_corrupted_nodes : int
        Number of corrupted (adversarial) nodes.
    max_delay : int
        Network delay bound Δ (in rounds).
    confirm_time : int
        Safety parameter k (confirmation time).
    probability : Optional[float], default None
        Per-round leader probability p. If None and c is provided, p = 1/(Δ·N·c).
        If None and c is None, fallback to 0.05.
    max_round : int, default 100
        Safety cap on the number of rounds.
    seed : Optional[int], default None
        Global seed passed to the runner.
    sigma : float, default 0.0
        Fraction of honest nodes sleeping/offline.
    c : Optional[float], default None
        Relative bandwidth/latency parameter used to derive p when probability is None.
    adversary_rush : bool, default True
        If True, adversary publishes at round+1; otherwise at round+Δ.
    delay_mode : str, default "worst"
        Delay policy: "worst" or "stochastic".
    delay_beta_a : float, default 2.0
        Beta(a, b) parameter (left shape) when delay_mode="stochastic".
    delay_beta_b : float, default 5.0
        Beta(a, b) parameter (right shape) when delay_mode="stochastic".
    auth_checks : bool, default False
        Enable signature verification and signing via TTP (FSign).
    enable_txs : bool, default False
        Enable transaction pool and transaction gossip.
    compact_broadcast : bool, default True
        Whether to use compact broadcast in the framework.

    Notes
    -----
    - p is clamped to (0, 1) with a tiny epsilon to avoid numerical edge cases.
    - Properties below expose a read-only view consistent with the framework.
    """

    def __init__(
        self,
        honest_node_type: Type[HonestNode],
        adversary_controller_type: Type[BalancingAttack],
        measurement_type: Type[ConsistencyMeasurement],
        num_honest_nodes: int,
        num_corrupted_nodes: int,
        max_delay: int,
        confirm_time: int,
        probability: Optional[float] = None,   # if None and c is set → computed
        max_round: int = 100,
        seed: Optional[int] = None,
        sigma: float = 0.0,                    # sleeping honest fraction
        c: Optional[float] = None,
        adversary_rush: bool = True,           # True: publish at round+1; False: at round+Δ
        delay_mode: str = "worst",             # "worst" / "stochastic"
        delay_beta_a: float = 2.0,
        delay_beta_b: float = 5.0,
        auth_checks: bool = False,
        enable_txs: bool = False,
        compact_broadcast: bool = True,
    ) -> None:
        # Concrete types
        self._honest_node_type = honest_node_type
        self._adversary_controller_type = adversary_controller_type
        self._measurement_type = measurement_type

        # Base parameters
        self._num_honest_nodes = int(num_honest_nodes)
        self._num_corrupted_nodes = int(num_corrupted_nodes)
        self._max_delay = int(max_delay)
        self._confirm_time = int(confirm_time)
        self._max_round = int(max_round)
        self._seed = seed
        self._sigma = float(sigma)
        self._c = None if c is None else float(c)
        self._adversary_rush = bool(adversary_rush)

        # Feature/behavior flags (were previously missing → could raise AttributeError)
        self._auth_checks = bool(auth_checks)
        self._enable_txs = bool(enable_txs)
        self._compact_broadcast = bool(compact_broadcast)

        # Derived / validated values
        total_nodes = self._num_honest_nodes + self._num_corrupted_nodes

        # Delay policy
        if delay_mode not in ("worst", "stochastic"):
            raise ValueError("delay_mode must be 'worst' or 'stochastic'")
        self._delay_mode = delay_mode
        self._delay_beta_a = float(delay_beta_a)
        self._delay_beta_b = float(delay_beta_b)

        # --- quick validations ---
        if total_nodes <= 0:
            raise ValueError("total_nodes must be > 0")
        if not (0.0 <= self._sigma < 1.0):
            raise ValueError("sigma must be in [0, 1)")
        if self._confirm_time < 0:
            raise ValueError("confirm_time must be >= 0")
        if self._confirm_time > self._max_round:
            raise ValueError("confirm_time cannot exceed max_round")
        if self._max_round <= 0:
            raise ValueError("max_round must be > 0")
        if self._max_delay <= 0:
            raise ValueError("max_delay must be > 0")

        # --- leader probability p ---
        if probability is None and self._c is not None:
            # p = 1 / (Δ · N · c)
            p = 1.0 / (self._max_delay * total_nodes * self._c)
        elif probability is None and self._c is None:
            # fallback compatible with older versions
            p = 0.05
        else:
            p = float(probability)

        # Clamp into (0,1) to avoid degenerate cases
        eps = 1e-12
        if p <= 0.0:
            p = eps
        elif p >= 1.0:
            p = 1.0 - eps
        self._probability = p

    # --- Properties required by the framework interface ----------------------

    @property
    def honest_node_type(self) -> Type[HonestNode]:
        return self._honest_node_type

    @property
    def adversary_controller_type(self) -> Type[BalancingAttack]:
        return self._adversary_controller_type

    @property
    def measurement_type(self) -> Type[ConsistencyMeasurement]:
        return self._measurement_type

    @property
    def num_honest_nodes(self) -> int:
        return self._num_honest_nodes

    @property
    def num_corrupted_nodes(self) -> int:
        return self._num_corrupted_nodes

    @property
    def max_delay(self) -> int:
        return self._max_delay

    @property
    def confirm_time(self) -> int:
        return self._confirm_time

    @property
    def probability(self) -> float:
        return self._probability

    @property
    def max_round(self) -> int:
        return self._max_round

    # Delay / adversary behavior
    @property
    def adversary_rush(self) -> bool:
        return self._adversary_rush

    @property
    def delay_mode(self) -> str:
        return self._delay_mode

    @property
    def delay_beta_a(self) -> float:
        return self._delay_beta_a

    @property
    def delay_beta_b(self) -> float:
        return self._delay_beta_b

    # Optional checks and transaction handling
    @property
    def auth_checks(self) -> bool:
        return self._auth_checks

    @property
    def enable_txs(self) -> bool:
        return self._enable_txs
