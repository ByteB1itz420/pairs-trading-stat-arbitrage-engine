"""Performance metrics and reproducible result exports."""

from pairs_trading.reporting.metrics import calculate_metrics
from pairs_trading.reporting.outputs import export_results, plot_backtest_report

__all__ = ["calculate_metrics", "export_results", "plot_backtest_report"]
