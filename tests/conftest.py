import pytest


@pytest.fixture
def mkt() -> dict:
    """Base-case market used across the test suite."""
    return {"s0": 100.0, "k": 105.0, "r": 0.03, "sigma": 0.2, "t": 1.0, "q": 0.01}
