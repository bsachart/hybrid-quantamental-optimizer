# Feature Specification: UI Simplification For Portfolio Workflow

**Feature Branch**: `002-ui-simplification`  
**Created**: 2026-05-05  
**Status**: Complete  
**Input**: User description: "Simplify the portfolio optimizer UI: reduce above-the-fold chrome, make the upload-to-solve path feel like one compact workflow, restructure results so users move cleanly from the risky portfolio to the target-volatility adjustment to the final allocation."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Follow One Clear Workflow (Priority: P1)

As an investor exploring the optimizer, I want the primary upload, solve, and
target-risk workflow to read as one clear sequence so that I can reach a
portfolio result without jumping between parallel panels that compete for
attention.

**Why this priority**: The app only delivers value after the user gets from raw
inputs to a solved portfolio. If that path feels dense or visually noisy, the
main product experience becomes harder to use.

**Independent Test**: This can be tested independently by opening the app,
uploading valid files, solving the portfolio, and confirming that the required
inputs, primary action, and next-step controls can be followed top-to-bottom
without relying on side-by-side panels.

**Acceptance Scenarios**:

1. **Given** a user opens the app for the first time, **When** they review the
   opening screen, **Then** they can identify the required input files,
   configuration controls, and solve action in one straightforward workflow.
2. **Given** valid price and metrics files are loaded, **When** the user solves
   the portfolio, **Then** the app transitions clearly from setup into results
   without forcing the user to rediscover where to act next.

---

### User Story 2 - Review Results In One Place (Priority: P2)

As an investor reviewing a solved portfolio, I want the tangency portfolio,
target-volatility adjustment, lending and borrowing controls, and Capital Market
Line to feel like one connected review surface so that I can understand the
result without piecing it together across separate sections.

**Why this priority**: A simplified setup flow still falls short if the result
section remains visually crowded or ambiguous about which numbers matter most.

**Independent Test**: This can be tested independently by solving the app with
valid inputs and confirming that a user can adjust target volatility, lending
rate, and borrowing rate from one nearby control cluster, then read the Capital
Market Line with clearly labelled lending and borrowing segments.

**Acceptance Scenarios**:

1. **Given** a solved risky portfolio, **When** the results section appears,
   **Then** the tangency metrics, target-risk control, lending and borrowing
   controls, and Capital Market Line are presented in a clearly ordered
   progression.
2. **Given** the user changes the target volatility, **When** the final
   portfolio updates, **Then** the app makes the lending-versus-borrowing
   position easy to interpret without leaving the results area.

---

### User Story 3 - Stay Oriented During Missing Inputs Or Errors (Priority: P3)

As an investor preparing data, I want missing-file states and optimization
failures to explain the next step plainly so that the simplified interface
still feels supportive when something is incomplete or invalid.

**Why this priority**: Simplification should remove clutter, not remove the
guidance users rely on when they are blocked.

**Independent Test**: This can be tested independently by opening the app with
missing files or invalid inputs and confirming that the interface still points
the user to the next corrective action.

**Acceptance Scenarios**:

1. **Given** one or both required files are missing, **When** the user reaches
   the setup area, **Then** the interface clearly states what is still needed
   before the portfolio can be solved.
2. **Given** optimization fails because the uploaded data is invalid, **When**
   the app shows the error, **Then** the message helps the user understand the
   problem without obscuring the controls needed to try again.

### Edge Cases

- What happens when only one of the two required files is uploaded?
- What happens when the user sets the target volatility to zero or to the full
  risky-portfolio volatility?
- What happens when the user sets the target volatility above the tangency
  portfolio and the borrowing rate exceeds the lending rate?
- What happens when optimization fails after the user has already reviewed the
  setup controls?
- What happens when the final allocation contains only cash or only risky
  assets?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The app MUST present the primary workflow as a clear sequence from
  file upload to model setup to solved portfolio.
- **FR-002**: The app MUST reduce non-essential visual and copy-heavy elements
  that compete with the primary upload and solve actions.
- **FR-003**: The app MUST keep sample-file guidance available without making it
  the dominant element of the default view.
- **FR-004**: The app MUST preserve the existing ability to upload price and
  metrics files, choose the risk model, set lending and borrowing rates, solve
  the risky portfolio, and adjust the final target volatility.
- **FR-005**: The app MUST present the tangency portfolio and the final
  cash-adjusted portfolio as distinct stages of the same workflow.
- **FR-006**: The app MUST preserve access to the current Capital Market Line
  visualization and result export path after the interface is simplified.
- **FR-007**: The app MUST provide plain-language status and error states for
  missing inputs, invalid uploads, and failed optimization attempts.
- **FR-008**: When target volatility exceeds the tangency portfolio, the system
  MUST support a leveraged borrowing segment of the Capital Market Line.
- **FR-009**: When the borrowing rate exceeds the lending rate, the system MUST
  show a kinked Capital Market Line at the tangency portfolio using common
  lending and borrowing terminology.

### Key Entities *(include if feature involves data)*

- **Workflow Stage**: A user-visible step in the portfolio journey such as file
  upload, model setup, solved risky portfolio, and final allocation review.
- **Portfolio Summary**: The set of headline metrics that describe either the
  risky portfolio or the final cash-adjusted portfolio.
- **Capital Market Line Segment**: The lending or borrowing portion of the
  line used to scale the tangency portfolio to the chosen target volatility.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a standard laptop-sized viewport, users can see the required
  inputs, model controls, and solve action without hunting across unrelated
  sections.
- **SC-002**: With valid input files, a first-time user can complete the
  upload-to-solve flow and reach a portfolio result in under 2 minutes.

## Assumptions

- The feature targets the existing browser-based optimizer experience rather
  than the Python API.
- The existing calculation behavior and data requirements remain the functional
  baseline for this UI change.
- Simplification focuses on layout, hierarchy, and messaging rather than adding
  new portfolio analytics.
- The current charts and downloadable results remain valuable and should stay
  available in the simplified experience.

## Out of Scope

- New optimization models or calculation changes
- New data import formats beyond the current CSV workflow
- User accounts, saved sessions, or portfolio history
- Full branding or marketing-site redesign outside the optimizer screen
- Single-column layout enforcement and control-group proximity (moved to spec 003)
- Merging final allocation summary and breakdown into one section (moved to spec 003)
