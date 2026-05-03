# Implementation Plan: Codebase Cleanup And Calculation Verification

**Branch**: `001-codebase-cleanup` | **Date**: 2026-05-03 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-codebase-cleanup/spec.md`

## Summary

This cleanup pass will protect calculation confidence first, then simplify the
engine at the seam with the highest duplicated portfolio math. The first slice
keeps the public portfolio API stable, expands deterministic verification around
portfolio scaling and covariance behavior, extracts shared portfolio-assembly
and capital-market-line math used by both engine entry points, and cleans up
repository noise that adds review churn without product value.

## Technical Context

**Language/Version**: Python 3.13.9  
**Primary Dependencies**: `numpy`, `scipy`, `polars`, `pandas`, `streamlit`, `pytest`  
**Storage**: CSV files and in-memory DataFrames  
**Testing**: `pytest` with co-located engine and script tests  
**Target Platform**: Local Python execution plus browser-oriented deployment path for the pandas engine  
**Project Type**: Python library with a Streamlit UI  
**Performance Goals**: Preserve current calculation behavior without material regression for current interactive and test workloads  
**Constraints**: Keep the public Python API stable, preserve both engine entry points, avoid broad architecture replacement in this slice  
**Scale/Scope**: One incremental cleanup focused on `src/engine`, deterministic verification, and lightweight repository hygiene

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Deep modules and simple interfaces**: Pass. Public entry points remain
  `optimize_portfolio`, `target_portfolio`, and `generate_cml`; duplicated math
  moves behind a smaller shared helper.
- **Small, atomic change**: Pass. This slice is limited to engine cleanup,
  verification, and repository hygiene rather than broad architectural churn.
- **Verification before completion**: Pass. The implementation will be gated by
  targeted `pytest` coverage for optimizer, risk, data loading, and portfolio
  orchestration behavior.
- **Professional clarity**: Pass. New shared helpers will use intention-revealing
  names and keep calculation invariants explicit.
- **High-leverage delivery**: Pass. The first slice targets the highest-risk
  duplicated calculation path before tackling deeper loader unification.
- **UI/UX guidance**: Not directly exercised in this slice because no user-facing
  UI behavior is planned. Any later UI changes remain subject to experienced UX
  and UI design judgment per the constitution.

## Project Structure

### Documentation (this feature)

```text
specs/001-codebase-cleanup/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── python-api.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── components/
├── engine/
│   ├── data_loader.py
│   ├── data_loader_pandas.py
│   ├── optimizer.py
│   ├── portfolio_engine.py
│   ├── portfolio_engine_pandas.py
│   ├── risk.py
│   ├── risk_pandas.py
│   └── *_test.py
├── scripts/
└── streamlit_app.py
```

**Structure Decision**: Keep the existing single-project layout and co-located
test structure. Add any new shared engine helper under `src/engine/` so both
the polars and pandas orchestration modules can use it without changing the
public API surface.

## Complexity Tracking

No constitution violations are planned for this slice.
