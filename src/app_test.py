import base64
import json

import pytest
import requests
from dash.testing.application_runners import ThreadedRunner

from src.app import (
    app,
    build_final_results,
    build_upload_status,
    parse_contents,
    solve_portfolio,
    toggle_solve_button,
)


PRICE_CSV = """date,AAPL,MSFT
2023-01-31,150.0,240.0
2023-02-28,155.0,245.0
2023-03-31,160.0,250.0
2023-04-30,162.0,255.0
"""

METRIC_CSV = """ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.10,0.20,0,1
MSFT,0.12,0.22,0,1
"""


def _encode_csv(csv_text: str) -> str:
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")
    return f"data:text/csv;base64,{encoded}"


def test_app_layout_contains_dash_workflow_controls():
    layout = str(app.layout)
    assert "upload-prices" in layout
    assert "upload-metrics" in layout
    assert "slider-target-vol" in layout
    assert "graph-cml" in layout


def test_parse_contents_reads_uploaded_csv():
    stream = parse_contents(_encode_csv(PRICE_CSV))
    assert "AAPL" in stream.getvalue()


def test_build_upload_status_returns_preview_and_store_payload():
    status, payload = build_upload_status(
        _encode_csv(METRIC_CSV),
        "metrics.csv",
        "Asset metrics",
    )

    assert status is not None
    assert payload["filename"] == "metrics.csv"
    assert payload["columns"] == [
        "ticker",
        "expected_return",
        "implied_volatility",
        "min_weight",
        "max_weight",
    ]
    assert payload["rows"] == 2


def test_toggle_solve_button_reflects_upload_readiness():
    disabled, waiting_message = toggle_solve_button(None, None)
    assert disabled is True
    assert "Waiting on files" in str(waiting_message)

    disabled, ready_message = toggle_solve_button({"text": PRICE_CSV}, {"text": METRIC_CSV})
    assert disabled is False
    assert "Ready to solve" in str(ready_message)


def test_solve_portfolio_returns_json_safe_state():
    solve_result = solve_portfolio(
        prices_payload={"text": PRICE_CSV},
        metrics_payload={"text": METRIC_CSV},
        risk_free_rate_percent=4.0,
        risk_model_value="FORWARD_LOOKING",
    )

    json.dumps(solve_result)

    tangency = solve_result["tangency"]
    assert tangency["tickers"] == ["aapl", "msft"]
    assert isinstance(tangency["weights"], list)
    assert solve_result["target_volatility_percent"] > 0


def test_build_final_results_handles_borrowing_floor():
    solve_result = solve_portfolio(
        prices_payload={"text": PRICE_CSV},
        metrics_payload={"text": METRIC_CSV},
        risk_free_rate_percent=4.0,
        risk_model_value="FORWARD_LOOKING",
    )

    results = build_final_results(
        tangency=solve_result["tangency"],
        target_volatility_percent=solve_result["target_volatility_percent"] * 1.5,
        borrowing_rate_percent=2.0,
        risk_free_rate_percent=4.0,
    )

    json.dumps(results["final_store"])
    assert "Borrowing rate adjusted" in str(results["final_metrics"])
    assert results["final_store"]["cash_weight"] < 0


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_dash_app_serves_html():
    with ThreadedRunner() as runner:
        runner.start(app)
        response = requests.get(runner.url, timeout=5)
        layout_response = requests.get(f"{runner.url.rstrip('/')}/_dash-layout", timeout=5)

    assert response.status_code == 200
    assert layout_response.status_code == 200
    assert "Hybrid Quantamental Optimizer" in response.text
    assert "Solve risky portfolio" in layout_response.text
