"""Per-market trading cost assumptions (deliberately conservative)."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CostModel:
    spread_bps: float          # Full bid/ask spread; half is paid on each side.
    commission_bps: float      # Per side.
    slippage_bps: float        # Per side.
    financing_bps_per_day: float = 0.0  # Holding cost on notional (swap / borrow).

    @property
    def per_side_frac(self) -> float:
        return (self.spread_bps / 2.0 + self.commission_bps + self.slippage_bps) / 1e4

    @property
    def financing_frac_per_day(self) -> float:
        return self.financing_bps_per_day / 1e4


DEFAULT_COSTS: Dict[str, CostModel] = {
    'forex': CostModel(spread_bps=1.5, commission_bps=0.0, slippage_bps=0.5, financing_bps_per_day=0.3),
    'metals': CostModel(spread_bps=6.0, commission_bps=0.0, slippage_bps=1.0, financing_bps_per_day=0.6),
    'crypto': CostModel(spread_bps=10.0, commission_bps=25.0, slippage_bps=5.0),
    'stocks': CostModel(spread_bps=2.0, commission_bps=0.0, slippage_bps=2.0),
}


def cost_for(asset_class: str) -> CostModel:
    return DEFAULT_COSTS.get(asset_class, DEFAULT_COSTS['stocks'])
