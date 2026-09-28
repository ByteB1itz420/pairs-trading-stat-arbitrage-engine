from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_visual_workbench_renders_and_runs_the_synthetic_example() -> None:
    app = streamlit_testing.AppTest.from_file(
        str(Path(__file__).parents[2] / "src" / "pairs_trading" / "dashboard.py"),
        default_timeout=60,
    ).run()

    assert not app.exception
    assert len(app.button) == 1
    app.button[0].click().run()

    assert not app.exception
    assert {metric.label for metric in app.metric} == {
        "Net return",
        "Gross return",
        "Maximum drawdown",
        "Trades",
    }
