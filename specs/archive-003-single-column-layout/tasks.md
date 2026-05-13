# Tasks: Single-Column Layout And Grouped Rate Controls

**Input**: Design documents from `/specs/003-single-column-layout/`
**Prerequisites**: plan.md, spec.md

**Tests**: Existing `pytest` suite in `src/components/results_display_test.py`
covers all pure-logic helpers and must remain green. Manual solve-flow
validation confirms the rendered layout.

**Organization**: Two focused tasks, sequentially dependent.

---

## Phase 1: Setup (Shared Context)

- [x] T001 Confirm existing tests pass before any changes: run `pytest src/`

---

## Phase 2: Implementation

### Task T002 — Remove two-column setup layout in `src/streamlit_app.py`

- [x] T002 [US1] Replace `st.columns([1.1, 0.9])` in the setup section with a
  sequential single-column stack. Containers for Step 1A, Step 1B, Step 2
  (assumptions), and the status+button card render top-to-bottom. No column
  wrappers remain in the setup flow.

**Files**: `src/streamlit_app.py`  
**Acceptance**: Setup section reads as a single vertical column; all upload,
rate, model, status, and solve controls remain functional.

---

### Task T003 — Consolidate CML controls and fix results layout

- [x] T003 [US2, US3, US4] In `src/streamlit_app.py`:
  - Merge Step 3B (target volatility) and Step 3D (lending/borrowing rates)
    into one container labelled "Capital Market Line controls" (new Step 3B).
  - Within that container: target volatility slider at full width, then
    lending rate display and borrowing rate slider on `st.columns(2)`.
  - Move the lending/borrowing status card into this same container, below
    the rate row.
  - Remove the first (stale) `target_portfolio()` call; compute once after
    the borrowing rate slider.
  - Remove the Step 3D container that previously followed Step 3C.

  In `src/components/results_display.py`:
  - Remove `st.columns([1.95, 0.95])` from `render_results()`.
  - Render CML chart at full width.
  - Move download button between the CML chart and the allocation section.
  - Merge "Final allocation" (bar chart) and "Final allocation breakdown"
    (table) into one unified section: heading → position summary → bar
    chart → table. Remove the separate "Final allocation breakdown" heading.

**Files**: `src/streamlit_app.py`, `src/components/results_display.py`  
**Acceptance**: Results read top-to-bottom with no lateral scanning required;
CML chart fills the full container width; allocation section is contiguous;
lending rate is visually distinct from the borrowing rate slider.

---

## Phase 3: Verification

- [x] T004 [P] Run `pytest src/` and confirm all tests pass
- [ ] T005 Run manual validation: launch app, load example files, solve, adjust
  target volatility and borrowing rate, confirm single-column reading order
  and unified allocation section

---

## Dependencies & Execution Order

- T001 → T002 → T003 → T004, T005
- T002 and T003 must run sequentially (both touch `src/streamlit_app.py`)
- T004 and T005 can run in parallel after T003

## Notes

- No engine or test logic changes are required.
- The existing `results_display_test.py` tests only pure-data helpers and
  will pass without modification.
- The `st.columns(2)` inside the CML controls container (for lending and
  borrowing rates) is intentional — it is a semantically paired comparison
  row, not a split reading sequence.
