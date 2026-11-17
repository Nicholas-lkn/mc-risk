"""Portfolio risk: VaR, CVaR, backtesting and stress scenarios (base currency CAD)."""

from mc_risk.risk.backtest import (
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    rolling_backtest,
    summarise_backtest,
)
from mc_risk.risk.data import (
    BENCHMARKS,
    DEFAULT_WEIGHTS,
    FX,
    PORTFOLIO,
    US_TICKERS,
    cad_returns,
    clean_prices,
    load_prices,
    synthetic_prices,
    to_cad,
)
from mc_risk.risk.stress import factor_shock, fx_shock, historical_replay
from mc_risk.risk.var import (
    VarResult,
    fit_student_t_dof,
    gaussian_mc_var,
    gaussian_var_closed_form,
    historical_var,
    student_t_mc_var,
    student_t_var_closed_form,
    var_cvar,
)

__all__ = [
    "BENCHMARKS",
    "DEFAULT_WEIGHTS",
    "FX",
    "PORTFOLIO",
    "US_TICKERS",
    "VarResult",
    "cad_returns",
    "christoffersen_independence",
    "clean_prices",
    "conditional_coverage",
    "factor_shock",
    "fit_student_t_dof",
    "fx_shock",
    "gaussian_mc_var",
    "gaussian_var_closed_form",
    "historical_replay",
    "historical_var",
    "kupiec_pof",
    "load_prices",
    "rolling_backtest",
    "student_t_mc_var",
    "student_t_var_closed_form",
    "summarise_backtest",
    "synthetic_prices",
    "to_cad",
    "var_cvar",
]
