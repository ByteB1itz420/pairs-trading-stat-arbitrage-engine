from io import StringIO

import pytest

from pairs_trading.dashboard import _load_uploaded_market


def test_upload_without_execution_open_price_explains_required_column() -> None:
    csv_file = StringIO(
        "timestamp,symbol,adjusted_close\n"
        "2025-01-01,AAA,100\n"
        "2025-01-01,BBB,90\n"
    )

    with pytest.raises(ValueError, match="required 'open' column"):
        _load_uploaded_market(csv_file, min_observations=1)


def test_upload_with_open_price_loads_and_reports_missing_cells() -> None:
    csv_file = StringIO(
        "timestamp,symbol,adjusted_close,open\n"
        "2025-01-01,AAA,100,99\n"
        "2025-01-01,BBB,90,89\n"
        "2025-01-02,AAA,101,100\n"
    )

    market = _load_uploaded_market(csv_file, min_observations=1)

    assert market.quality.symbols == ("AAA", "BBB")
    assert market.quality.missing_open == {"AAA": 0, "BBB": 1}
