from src.components.workflow_state import (
    build_run_summary,
    build_setup_status,
    build_workflow_stages,
)


def test_build_setup_status_requires_both_files():
    status = build_setup_status(None, None)

    assert status.title == "Upload both files"
    assert status.tone == "neutral"
    assert status.ready_to_solve is False


def test_build_setup_status_handles_single_missing_file():
    price_only = build_setup_status("prices.csv", None)
    metrics_only = build_setup_status(None, "metrics.csv")

    assert price_only.title == "Add the metrics file"
    assert price_only.ready_to_solve is False
    assert metrics_only.title == "Add the price history file"
    assert metrics_only.ready_to_solve is False


def test_build_setup_status_ready_to_solve():
    status = build_setup_status("prices.csv", "metrics.csv")

    assert status.title == "Ready to solve"
    assert status.tone == "ready"
    assert status.ready_to_solve is True
    assert "prices.csv and metrics.csv" in status.message


def test_build_setup_status_handles_completed_solve():
    status = build_setup_status(
        "prices.csv",
        "metrics.csv",
        optimization_complete=True,
    )

    assert status.title == "Portfolio solved"
    assert status.tone == "success"
    assert status.ready_to_solve is True


def test_build_setup_status_handles_errors_without_losing_readiness():
    status = build_setup_status(
        "prices.csv",
        "metrics.csv",
        error_message="Ticker mismatch.",
    )

    assert status.title == "Solve failed"
    assert status.tone == "error"
    assert status.ready_to_solve is True
    assert "Ticker mismatch." in status.message


def test_build_workflow_stages_default_state():
    stages = build_workflow_stages(
        ready_to_solve=False,
        optimization_complete=False,
    )

    assert [stage.status for stage in stages] == ["active", "pending", "pending"]


def test_build_workflow_stages_ready_state():
    stages = build_workflow_stages(
        ready_to_solve=True,
        optimization_complete=False,
    )

    assert [stage.status for stage in stages] == ["complete", "active", "pending"]


def test_build_workflow_stages_solved_state():
    stages = build_workflow_stages(
        ready_to_solve=True,
        optimization_complete=True,
    )

    assert [stage.status for stage in stages] == ["complete", "complete", "active"]


def test_build_run_summary_formats_current_inputs():
    summary = build_run_summary(
        "prices.csv",
        "metrics.csv",
        lending_rate=0.04,
        borrowing_rate=0.06,
        risk_model_label="Forward-Looking",
    )

    assert summary == (
        "prices.csv + metrics.csv · Forward-Looking · lending 4.00% · borrowing 6.00%"
    )


def test_build_run_summary_collapses_equal_rates():
    summary = build_run_summary(
        "prices.csv",
        "metrics.csv",
        lending_rate=0.04,
        borrowing_rate=0.04,
        risk_model_label="Forward-Looking",
    )

    assert summary == "prices.csv + metrics.csv · Forward-Looking · rate 4.00%"
