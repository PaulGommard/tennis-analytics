"""Swappable tennis data providers."""

from tennis_analytics.product.providers.base import TennisDataProvider
from tennis_analytics.product.providers.historical import HistoricalTennisDataProvider

__all__ = ["HistoricalTennisDataProvider", "TennisDataProvider"]
