# Implementation Plan: UI Theme Support & Integrated Testing

**Branch**: `003-ui-theme-testing` | **Date**: 2026-05-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-ui-theme-testing/spec.md`

## Summary

This feature implements robust light and dark mode support for the Portfolio Optimizer and introduces integrated Streamlit testing. The technical approach centers on a CSS-variable-driven design system that responds to system theme changes and Altair chart configurations that adapt to the background. Integrated testing will use Streamlit's `AppTest` framework to verify the end-to-end "upload-to-solve" journey.

## Technical Context

**Language/Version**: Python 3.13.9  
**Primary Dependencies**: `streamlit` (1.52.1), `altair`, `pandas`, `pytest`  
**Storage**: N/A (In-memory session state)  
**Testing**: `pytest` with `streamlit.testing.v1.AppTest`  
**Target Platform**: Local Streamlit execution  
**Project Type**: Python Web App (Streamlit)  
**Performance Goals**: Instant theme switching; <2s for automated app tests  
**Constraints**: Must maintain "premium" aesthetics; avoid generic/default looks  
**Scale/Scope**: Main app (`streamlit_app.py`) and Results component (`results_display.py`)

## Constitution Check

- **Deep modules and simple interfaces**: Pass. Theme logic is hidden in `_inject_styles`; tests expose a clean "run and verify" interface.
- **Small, atomic change**: Pass. Focuses specifically on theming and testing the existing UI flow.
- **Verification before completion**: Pass. New `AppTest` suite will provide automated verification.
- **Professional clarity**: Pass. Uses semantic CSS variables and clear test scenarios.
- **High-leverage delivery**: Pass. Fixes the "broken" UI reported by the user while adding a safety net for future changes.

## Project Structure

### Documentation (this feature)

```text
specs/003-ui-theme-testing/
├── plan.md
├── research.md
├── checklists/
│   └── requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── components/
│   ├── results_display.py      # Updated for theme-aware Altair/CSS
│   └── ...
├── streamlit_app.py            # Updated for CSS variable injection
└── streamlit_app_test.py      # NEW: Integrated AppTests
```

**Structure Decision**: Co-locate the app-level test (`streamlit_app_test.py`) with the app itself, following the project's locality principle.

## Complexity Tracking

No constitution violations are planned for this slice.
