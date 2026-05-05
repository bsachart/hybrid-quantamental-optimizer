# Feature Specification: UI Simplification For Portfolio Workflow

**Feature Branch**: `002-ui-simplification`  
**Created**: 2026-05-05  
**Status**: Draft  
**Input**: User description: "Avoid two columns whenever possible, merge final allocation with final allocation breakdown, keep lending and borrowing rates separate, and move target volatility plus lending/borrowing controls closer together."

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
   without forcing the user to rediscover where to act next or scan across
   separate columns for the primary next action.

---

### User Story 2 - Review Results In One Place (Priority: P2)

As an investor reviewing a solved portfolio, I want the tangency portfolio,
target-volatility adjustment, lending and borrowing controls, Capital Market
Line, and final allocation to feel like one connected review surface so that I
can understand the result without piecing it together across separate side-by-
side sections.

**Why this priority**: A simplified setup flow still falls short if the result
section remains visually crowded or ambiguous about which numbers matter most.

**Independent Test**: This can be tested independently by solving the app with
valid inputs and confirming that a user can adjust target volatility, lending
rate, and borrowing rate from one nearby control cluster, then read the final
allocation and its breakdown from one contiguous section.

**Acceptance Scenarios**:

1. **Given** a solved risky portfolio, **When** the results section appears,
   **Then** the tangency metrics, target-risk control, lending and borrowing
   controls, Capital Market Line, and final allocation are presented in a
   clearly ordered progression.
2. **Given** the user changes the target volatility, **When** the final
   portfolio updates, **Then** the app makes the lending-versus-borrowing
   position and the resulting allocation easy to interpret without leaving the
   results area.
3. **Given** the user reviews the final portfolio, **When** they inspect the
   allocation section, **Then** the headline allocation view and the detailed
   allocation breakdown appear as one unified section rather than separate
   panels.

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
- What happens when the screen is wide enough for multiple columns but the task
  is still easier to follow in a single reading order?

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
- **FR-006**: The app MUST keep target volatility, lending rate, and borrowing
  rate controls physically close to the Capital Market Line so users can see
  cause and effect without scanning across distant sections.
- **FR-007**: The app MUST place lending and borrowing rate controls on the
  same row or within the same compact control group.
- **FR-008**: The app MUST avoid two-column layouts when a single reading order
  better supports comprehension, especially for the primary workflow and the
  result interpretation flow.
- **FR-009**: The app MUST merge the final allocation summary and the final
  allocation breakdown into one unified allocation section.
- **FR-010**: The app MUST preserve access to the current Capital Market Line
  visualization and result export path after the interface is simplified.
- **FR-011**: The app MUST provide plain-language status and error states for
  missing inputs, invalid uploads, and failed optimization attempts.
- **FR-012**: When target volatility exceeds the tangency portfolio, the system
  MUST support a leveraged borrowing segment of the Capital Market Line.
- **FR-013**: When the borrowing rate exceeds the lending rate, the system MUST
  show a kinked Capital Market Line at the tangency portfolio using common
  lending and borrowing terminology.

### Key Entities *(include if feature involves data)*

- **Workflow Stage**: A user-visible step in the portfolio journey such as file
  upload, model setup, solved risky portfolio, and final allocation review.
- **Control Cluster**: The group of closely related user controls for target
  volatility, lending rate, and borrowing rate.
- **Portfolio Summary**: The set of headline metrics that describe either the
  risky portfolio or the final cash-adjusted portfolio.
- **Allocation View**: The user-facing breakdown that shows how much capital is
  assigned to each risky asset and how much remains in cash or borrowing.
- **Capital Market Line Segment**: The lending or borrowing portion of the
  line used to scale the tangency portfolio to the chosen target volatility.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a standard laptop-sized viewport, users can see the required
  inputs, model controls, and solve action without hunting across unrelated
  sections.
- **SC-002**: With valid input files, a first-time user can complete the
  upload-to-solve flow and reach a portfolio result in under 2 minutes.
- **SC-003**: After solving, users can find target volatility plus lending and
  borrowing controls within one nearby control area and understand that they
  govern the Capital Market Line.
- **SC-004**: Users can locate the final allocation summary and detailed
  allocation breakdown within one contiguous section in under 10 seconds.

## Assumptions

- The feature targets the existing browser-based optimizer experience rather
  than the Python API.
- The existing calculation behavior and data requirements remain the functional
  baseline for this UI change.
- Simplification focuses on layout, hierarchy, and messaging rather than adding
  new portfolio analytics.
- The current charts and downloadable results remain valuable and should stay
  available in the simplified experience.
- A stacked or mostly single-column reading flow is preferred unless side-by-
  side comparison clearly improves the task.

## Out of Scope

- New optimization models or calculation changes
- New data import formats beyond the current CSV workflow
- User accounts, saved sessions, or portfolio history
- Full branding or marketing-site redesign outside the optimizer screen
