"""mc-risk: Monte Carlo option pricing and portfolio risk."""

from mc_risk.engine import mc_price, price_european
from mc_risk.result import PriceResult

__all__ = ["PriceResult", "mc_price", "price_european"]
__version__ = "0.1.0"
