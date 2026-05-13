from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest
import pytest

def test_app_smoke_test():
    """Verify the app loads and contains the header."""
    at = AppTest.from_file("src/streamlit_app.py").run()
    assert not at.exception
    assert any("Hybrid Quantamental Optimizer" in m.value for m in at.markdown)

@patch("streamlit.file_uploader")
def test_app_upload_and_solve_flow(mock_uploader):
    """Verify the happy path flow using mocking for file uploaders."""
    # Prepare mock data
    price_csv = "date,AAPL,MSFT\n2023-01-31,150.0,200.0\n2023-02-28,155.0,210.0\n2023-03-31,160.0,220.0"
    metric_csv = "ticker,expected_return,implied_volatility,min_weight,max_weight\nAAPL,0.1,0.2,0,1\nMSFT,0.12,0.22,0,1"
    
    price_file = MagicMock()
    price_file.name = "prices.csv"
    price_file.getvalue.return_value = price_csv.encode("utf-8")
    price_file.seek = MagicMock()
    
    metric_file = MagicMock()
    metric_file.name = "metrics.csv"
    metric_file.getvalue.return_value = metric_csv.encode("utf-8")
    metric_file.seek = MagicMock()
    
    # We need to handle multiple calls across reruns.
    def uploader_side_effect(label, *args, **kwargs):
        if "prices" in label.lower():
            return price_file
        return metric_file
    
    mock_uploader.side_effect = uploader_side_effect
    
    at = AppTest.from_file("src/streamlit_app.py").run()
    
    # Check that it's ready to solve
    assert any("Ready to solve" in m.value for m in at.markdown)
    
    # Click Solve
    # Note: clicking the button in AppTest triggers a rerun
    at.button(key="solve_button").click().run()
    
    # Verify success with diagnostics
    if "solve_error" in at.session_state and at.session_state.solve_error:
        pytest.fail(f"Optimization failed with error: {at.session_state.solve_error}")
        
    assert at.session_state.optimization_complete is True
    assert any("Results" in m.value for m in at.markdown)
