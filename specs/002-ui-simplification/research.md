# Research: UI Simplification For Portfolio Workflow

## Decision: Replace the large intro treatment with a compact workflow header

- **Rationale**: The current hero plus secondary explainer consumes the most
  prominent screen space before the user reaches the actual controls. A compact
  header keeps orientation while letting users reach the upload and solve path
  faster.
- **Alternatives considered**:
  - Keep the current hero and shorten the copy. Rejected because the layout
    overhead would still dominate the opening viewport.
  - Split the experience into multiple pages. Rejected because it would add
    navigation complexity to a flow that should stay linear.

## Decision: Keep file guidance available on demand inside the upload area

- **Rationale**: Sample formats are important for first-time users, but they do
  not need to compete with the default workflow. Making them secondary preserves
  help without crowding the initial scan path.
- **Alternatives considered**:
  - Remove sample guidance entirely. Rejected because it would make blocked
    users less self-sufficient.
  - Show full sample data by default. Rejected because it adds noise before the
    user has even chosen files.

## Decision: Present results as a narrative sequence from risky mix to final mix

- **Rationale**: Users need to understand that the tangency portfolio is the
  intermediate risky mix and the final allocation is the cash-adjusted outcome.
  A sequential layout makes the two-stage methodology easier to follow.
- **Alternatives considered**:
  - Keep all result panels visually equal. Rejected because it makes the
    relationship between tangency and final allocation harder to read.
  - Hide the tangency portfolio after solving. Rejected because it removes a
    meaningful part of the methodology and weakens interpretability.

## Decision: Extract only lightweight workflow-state helpers

- **Rationale**: The app benefits from a small amount of structure around
  status messaging and stage labels, but a broader component framework would add
  ceremony without enough leverage for a single-screen workflow.
- **Alternatives considered**:
  - Keep all logic inline in `streamlit_app.py`. Rejected because the UI changes
    would continue to grow an already dense file.
  - Introduce a larger view-model or page framework. Rejected because it would
    exceed the scope of a focused simplification slice.
