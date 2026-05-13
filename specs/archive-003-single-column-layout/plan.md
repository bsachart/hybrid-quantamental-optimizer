# Implementation Plan: Single-Column Layout And Grouped Rate Controls

**Branch**: `003-single-column-layout` | **Date**: 2026-05-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-single-column-layout/spec.md`

## Summary

Two surgical changes to the existing UI layer. No engine or API changes.

1. **Setup section** — replace the side-by-side upload/solve columns with a
   single vertical stack of containers.
2. **Results section** — consolidate the three-part results layout (CML chart |
   allocation chart + summary || breakdown table) into a clear top-to-bottom
   sequence: CML controls cluster → final portfolio metrics → CML chart →
   unified allocation section. Remove all `st.columns` that split a logical
   reading sequence.

Both changes are purely presentational. All calculations, chart logic, and
export behavior remain byte-for-byte identical.

## Technical Context

**Language/Version**: Python 3.13.9  
**Primary Dependencies**: `streamlit`, `altair`, `pandas`, `pytest`  
**Storage**: In-memory Streamlit session state  
**Testing**: `pytest` with co-located test files; manual `streamlit run` validation  
**Target Platform**: Local Streamlit execution and Stlite deployment via `src/index.html`  
**Project Type**: Python library with a Streamlit UI  
**Performance Goals**: No regression — all changes are layout-only  
**Constraints**: No engine changes, no new dependencies, single-screen flow preserved  
**Scale/Scope**: Two files — `src/streamlit_app.py` and `src/components/results_display.py`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Deep modules and simple interfaces**: Pass. No new abstractions introduced;
  existing rendering helpers are reused in a different order.
- **Small, atomic change**: Pass. Scope is two files, layout-only.
- **Verification before completion**: Pass. Existing `pytest` suite covers all
  pure-logic helpers; manual solve-flow validation confirms rendering.
- **Professional clarity**: Pass. The change enforces cleaner visual hierarchy
  and removes ambiguous parallel reading streams.
- **High-leverage delivery**: Pass. Addresses the highest remaining UX friction
  in the results flow without touching the calculation engine.
- **UI/UX guidance**: Pass. Changes apply Gestalt proximity (grouped controls),
  single reading direction, and information consolidation principles.

## Project Structure

### Documentation (this feature)

```text
specs/003-single-column-layout/
├── plan.md
├── tasks.md
└── checklists/
    └── requirements.md
```

### Source Code (repository root)

```text
src/
├── components/
│   ├── results_display.py          ← remove two-column split, merge allocation
│   └── results_display_test.py    ← no changes needed
└── streamlit_app.py               ← remove setup columns, consolidate 3B+3D
```

## Key Design Decisions

### Setup: true single column

Replace `st.columns([1.1, 0.9])` with sequential containers. Upload containers
(1A, 1B) appear before the solve container (Step 2) and the status+button
container. This is the natural reading order and removes the "where do I look
next?" problem.

### CML Controls: merged Step 3B + 3D

Target volatility slider and the lending/borrowing rate row move into one
container labelled "Capital Market Line controls". The borrowing rate slider
now appears before the final portfolio computation, eliminating the two-pass
`target_portfolio()` call that used stale session state for Step 3C metrics.
This is a correctness improvement as well as a layout improvement.

The lending rate and borrowing rate appear on `st.columns(2)` within this
container — two related but distinct values on one row. This is not a
"split reading sequence" column; it is a semantically paired comparison.

### Results: sequential, unified allocation

`render_results()` loses its `st.columns([1.95, 0.95])` split. CML chart
goes full width. Allocation section becomes one contiguous block: heading →
position summary → bar chart → table. Download button moves between the CML
chart and the allocation section.

## Complexity Tracking

No constitution violations are planned for this slice.
