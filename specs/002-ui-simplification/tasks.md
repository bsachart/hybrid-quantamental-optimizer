# Tasks: UI Simplification For Portfolio Workflow

**Input**: Design documents from `/specs/002-ui-simplification/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ui-workflow.md, quickstart.md

**Tests**: Focused verification is required for this feature. Add or extend
co-located `pytest` coverage for any extracted UI workflow helpers, then run the
manual Streamlit workflow checks from `quickstart.md`.

**Organization**: Tasks are grouped by user story so each story can be
implemented and validated independently where practical.

## Phase 1: Setup (Shared Context)

**Purpose**: Confirm the active workflow, fixtures, and verification entry
points before changing the UI

- [x] T001 Confirm `examples/sample_prices.csv`, `examples/sample_metrics.csv`, and `specs/002-ui-simplification/quickstart.md` match the intended manual validation flow

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Introduce shared workflow-state building blocks before the main UI
layout changes

**⚠️ CRITICAL**: No user story work should start until this phase is complete

- [x] T002 Create shared workflow-state helpers in `src/components/workflow_state.py`
- [x] T003 Add co-located tests for workflow-state helpers in `src/components/workflow_state_test.py`
- [x] T004 Refactor shared stage/status rendering helpers in `src/streamlit_app.py` to consume `src/components/workflow_state.py`

**Checkpoint**: Workflow status and stage metadata are centralized and testable

---

## Phase 3: User Story 1 - Complete The Workflow Without Friction (Priority: P1) 🎯 MVP

**Goal**: Make the upload, model-setup, and solve path feel like one compact
workflow

**Independent Test**: Launch the app, keep the screen in its default state,
load both example files, and confirm the required inputs and solve action stay
easy to locate in a single workflow.

### Verification for User Story 1

- [x] T005 [P] [US1] Extend workflow readiness and stage-order tests in `src/components/workflow_state_test.py`

### Implementation for User Story 1

- [x] T006 [US1] Replace the large intro treatment with a compact workflow header in `src/streamlit_app.py`
- [x] T007 [US1] Simplify the upload area and make sample-format help secondary in `src/streamlit_app.py`
- [x] T008 [US1] Consolidate model controls, readiness messaging, and the primary solve action in `src/streamlit_app.py`

**Checkpoint**: The user can move from open screen to ready-to-solve state
without scanning through extra interface chrome

---

## Phase 4: User Story 2 - Read The Output At A Glance (Priority: P2)

**Goal**: Clarify the relationship between the risky portfolio, target
volatility, and final allocation

**Independent Test**: Solve the portfolio with the example files and confirm
the results read in order from tangency summary to target-volatility control to
final allocation analysis.

### Verification for User Story 2

- [x] T009 [P] [US2] Add result-summary and borrowing-segment coverage in `src/components/results_display_test.py` and `src/engine/portfolio_math_test.py`

### Implementation for User Story 2

- [x] T010 [US2] Implement lending and borrowing Capital Market Line scaling in `src/engine/portfolio_math.py`, `src/engine/portfolio_engine.py`, `src/engine/portfolio_engine_pandas.py`, `src/engine/portfolio_math_test.py`, `src/engine/portfolio_engine_test.py`, and `src/engine/portfolio_engine_pandas_test.py`
- [x] T011 [US2] Reorganize the solved-state summary flow and Capital Market Line terminology in `src/streamlit_app.py` and `src/components/results_display.py`
- [x] T012 [US2] Keep final-allocation metrics, charts, and export behavior aligned after target changes in `src/streamlit_app.py`, `src/components/results_display.py`, and `src/components/results_display_test.py`

**Checkpoint**: A solved portfolio clearly communicates what the risky mix is,
how target volatility affects it, and what the final allocation becomes

---

## Phase 5: User Story 3 - Stay Oriented During Missing Inputs Or Errors (Priority: P3)

**Goal**: Preserve supportive guidance while removing unnecessary clutter

**Independent Test**: Open the app with missing files and then with an invalid
input scenario, and confirm the interface still points to the next corrective
step without hiding the controls.

### Verification for User Story 3

- [x] T013 [P] [US3] Extend workflow-state tests for missing-input and error messaging in `src/components/workflow_state_test.py`

### Implementation for User Story 3

- [x] T014 [US3] Improve missing-input status copy and next-step guidance in `src/streamlit_app.py`
- [x] T015 [US3] Preserve recoverable solve-error messaging and retry flow in `src/streamlit_app.py`

**Checkpoint**: Missing-file and failed-solve states stay clear, compact, and
recoverable

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification and cleanup across the simplified workflow

- [x] T016 [P] Tighten shared copy and visual labels across `src/streamlit_app.py` and `src/components/results_display.py`
- [x] T017 Run `pytest src/components/workflow_state_test.py src/components/results_display_test.py src/engine/portfolio_engine_pandas_test.py src/engine/portfolio_math_test.py`
- [ ] T018 Run the manual validation flow documented in `specs/002-ui-simplification/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup and blocks all story work
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on User Story 1’s streamlined layout baseline
- **User Story 3 (Phase 5)**: Depends on Foundational completion and should land after the main workflow copy/layout changes
- **Polish (Phase 6)**: Depends on all targeted stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Establishes the simplified primary workflow and is the MVP
- **User Story 2 (P2)**: Builds on the solved-state layout introduced in US1
- **User Story 3 (P3)**: Reuses the workflow-state helpers and final messaging structure from earlier phases

### Within Each User Story

- Verification tasks should land before or alongside the corresponding UI logic
- Shared helper changes should precede layout rewrites
- Copy and state handling should stay consistent with the workflow contract

### Parallel Opportunities

- `T003` can run in parallel with planning-level review once `T002` defines the helper surface
- `T005` and `T013` can run independently from most layout edits because they target the shared workflow helper
- `T009` can run in parallel with solved-state layout updates because it targets `src/components/results_display_test.py`
- `T016` can run after all story behavior is stable

## Parallel Example: User Story 1

```bash
Task: "Extend workflow readiness and stage-order tests in src/components/workflow_state_test.py"
Task: "Simplify the upload area and make sample-format help secondary in src/streamlit_app.py"
```

## Parallel Example: User Story 2

```bash
Task: "Add result-summary helper coverage for allocation and export behavior in src/components/results_display_test.py"
Task: "Simplify chart, allocation, and readout presentation in src/components/results_display.py"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Deliver User Story 1.
3. Validate the compact upload-to-solve workflow manually.

### Incremental Delivery

1. Centralize workflow-state handling.
2. Simplify the setup workflow.
3. Clarify solved-state interpretation.
4. Tighten missing-input and error recovery states.
5. Run automated and manual verification.

## Notes

- Tasks include focused test work because the plan and constitution require
  reproducible verification for the simplified UI flow.
- Avoid changing engine calculations or public API behavior while working this
  task list.
