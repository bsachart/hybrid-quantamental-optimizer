"""Pure helpers for the Streamlit workflow state and stage messaging."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkflowStatus:
    """User-facing setup status shown beside the solver controls."""

    title: str
    message: str
    tone: str
    ready_to_solve: bool


@dataclass(frozen=True, slots=True)
class WorkflowStage:
    """A single step in the portfolio workflow."""

    name: str
    title: str
    description: str
    status: str


def build_setup_status(
    prices_name: str | None,
    metrics_name: str | None,
    *,
    optimization_complete: bool = False,
    error_message: str | None = None,
) -> WorkflowStatus:
    """Build the current setup-state message for the UI."""
    ready_to_solve = bool(prices_name and metrics_name)

    if error_message:
        return WorkflowStatus(
            title="Solve failed",
            message=(
                f"{error_message} Check the uploaded files and assumptions, then try again."
            ),
            tone="error",
            ready_to_solve=ready_to_solve,
        )

    if optimization_complete and ready_to_solve:
        return WorkflowStatus(
            title="Portfolio solved",
            message=(
                "Review the risky portfolio, choose the target volatility, and inspect "
                "the final allocation below."
            ),
            tone="success",
            ready_to_solve=True,
        )

    if ready_to_solve:
        return WorkflowStatus(
            title="Ready to solve",
            message=(
                f"{prices_name} and {metrics_name} are loaded. Set the assumptions "
                "and solve the risky portfolio."
            ),
            tone="ready",
            ready_to_solve=True,
        )

    if prices_name:
        return WorkflowStatus(
            title="Add the metrics file",
            message=(
                f"{prices_name} is loaded. Upload the asset metrics CSV to continue."
            ),
            tone="warning",
            ready_to_solve=False,
        )

    if metrics_name:
        return WorkflowStatus(
            title="Add the price history file",
            message=(
                f"{metrics_name} is loaded. Upload the price history CSV to continue."
            ),
            tone="warning",
            ready_to_solve=False,
        )

    return WorkflowStatus(
        title="Upload both files",
        message=(
            "Add a price history CSV and an asset metrics CSV to unlock the solver."
        ),
        tone="neutral",
        ready_to_solve=False,
    )


def build_workflow_stages(
    *,
    ready_to_solve: bool,
    optimization_complete: bool,
) -> tuple[WorkflowStage, WorkflowStage, WorkflowStage]:
    """Build the ordered workflow stages for the simplified screen."""
    upload_status = "complete" if ready_to_solve else "active"
    solve_status = (
        "complete" if optimization_complete else "active" if ready_to_solve else "pending"
    )
    results_status = "active" if optimization_complete else "pending"

    return (
        WorkflowStage(
            name="upload",
            title="Upload data",
            description="Add the price history and asset metrics files.",
            status=upload_status,
        ),
        WorkflowStage(
            name="solve",
            title="Solve risky mix",
            description="Choose assumptions and build the tangency portfolio.",
            status=solve_status,
        ),
        WorkflowStage(
            name="results",
            title="Shape final allocation",
            description="Set target volatility and review the final mix.",
            status=results_status,
        ),
    )


def build_run_summary(
    prices_name: str,
    metrics_name: str,
    *,
    lending_rate: float,
    borrowing_rate: float,
    risk_model_label: str,
) -> str:
    """Summarize the current solve inputs in one short line."""
    if abs(borrowing_rate - lending_rate) <= 1e-12:
        rate_summary = f"rate {lending_rate:.2%}"
    else:
        rate_summary = (
            f"lending {lending_rate:.2%} · borrowing {borrowing_rate:.2%}"
        )

    return (
        f"{prices_name} + {metrics_name} · {risk_model_label} · {rate_summary}"
    )
