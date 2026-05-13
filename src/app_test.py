import pytest
from src.app import app, parse_contents
import io
import base64

def test_app_layout():
    """Verify the app layout is initialized correctly."""
    assert app.layout is not None
    # Check for core components
    layout_str = str(app.layout)
    assert "upload-prices" in layout_str
    assert "upload-metrics" in layout_str
    assert "btn-solve" in layout_str

def test_parse_contents():
    """Verify CSV content parsing."""
    csv_data = "date,AAPL\n2023-01-31,150.0"
    base64_data = base64.b64encode(csv_data.encode("utf-8")).decode("utf-8")
    contents = f"data:text/csv;base64,{base64_data}"
    
    stream = parse_contents(contents)
    import pandas as pd
    df = pd.read_csv(stream)
    assert not df.empty
    assert "AAPL" in df.columns
