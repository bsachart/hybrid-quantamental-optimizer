# Implementation Plan: UI Simplification For Portfolio Workflow

**Branch**: `002-ui-simplification` | **Date**: 2026-05-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-ui-simplification/spec.md`

## Summary

This slice simplifies the browser workflow for the portfolio optimizer by
reducing above-the-fold chrome, keeping upload and solve controls visible
together, and restructuring the results flow so users move cleanly from the
risky portfolio to the target-volatility adjustment to the final allocation.
The calculation engine, charts, and export behavior remain intact; any new
abstractions stay lightweight and focused on UI workflow clarity.

## Technical Context

**Language/Version**: Python 3.13.9  
**Primary Dependencies**: `streamlit`, `pandas`, `numpy`, `scipy`, `altair`, `pytest`  
**Storage**: CSV uploads plus in-memory Streamlit session state  
**Testing**: `pytest` with co-located test files plus manual `streamlit run` workflow validation  
**Target Platform**: Local Streamlit execution and browser-oriented Stlite deployment via `src/index.html`  
**Project Type**: Python library with a Streamlit UI  
**Performance Goals**: Preserve current solve responsiveness for sample-sized CSV inputs and keep target-volatility adjustments immediate once a portfolio is solved  
**Constraints**: Keep calculation behavior unchanged, preserve current CSV inputs and result export, keep the interface readable on desktop and narrow screens, avoid turning the workflow into a multi-page flow  
**Scale/Scope**: One Streamlit screen, one results component, and lightweight UI helper/test additions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Deep modules and simple interfaces**: Pass. Any new abstraction is limited
  to lightweight workflow-state helpers that reduce `streamlit_app.py`
  complexity without widening the public surface.
- **Small, atomic change**: Pass. Scope is limited to the optimizer UI flow,
  not engine calculations, API changes, or broader site redesign.
- **Verification before completion**: Pass. Work will be checked with focused
  `pytest` coverage for extracted UI helpers plus manual solve-flow validation.
- **Professional clarity**: Pass. The feature explicitly targets clearer
  labels, hierarchy, and stage transitions in the app.
- **High-leverage delivery**: Pass. This slice focuses on the highest-friction
  screen in the product and preserves existing calculations.
- **UI/UX guidance**: Pass. The plan centers on clearer user flow, meaningful
  states, accessible messaging, and more intentional visual hierarchy.

## Project Structure

### Documentation (this feature)

```text
specs/002-ui-simplification/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── ui-workflow.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── components/
│   ├── results_display.py
│   ├── workflow_state.py
│   └── workflow_state_test.py
├── engine/
│   └── ...
├── index.html
└── streamlit_app.py
```

**Structure Decision**: Keep the existing single-project layout and co-located
test style. Limit the implementation to `src/streamlit_app.py`,
`src/components/results_display.py`, and a small `src/components/workflow_state.py`
helper so UI behavior stays readable and directly testable without changing the
engine modules.

## Complexity Tracking

No constitution violations are planned for this slice.
