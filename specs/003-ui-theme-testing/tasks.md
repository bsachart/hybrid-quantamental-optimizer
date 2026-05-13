# Tasks: UI Theme Support & Integrated Testing

## Phase 1: CSS & Theme Refactor

- [ ] **Task 1: Define Semantic CSS Variables**
  - Extract all hardcoded hex values in `src/streamlit_app.py` into CSS variables.
  - Create a `@media (prefers-color-scheme: dark)` block for dark mode overrides.
  - Verify variables work with the existing layout.
- [ ] **Task 2: Update Results Component Styling**
  - Update `src/components/results_display.py` to use CSS variables for the allocation table.
  - Make Altair charts theme-aware (transparent background, dynamic axis colors).
- [ ] **Task 3: Fix Visual Glitches**
  - Audit contrast for buttons, status cards, and file uploaders in both modes.
  - Fix any white-on-white or dark-on-dark issues.

## Phase 2: Integrated Testing

- [ ] **Task 4: Implement Basic AppTest Scaffolding**
  - Create `src/streamlit_app_test.py`.
  - Implement a basic test that loads the app and checks for the title.
- [ ] **Task 5: Implement "Happy Path" Integration Test**
  - Mock price and metric CSV data.
  - Test file upload -> Solve button enablement -> Solve execution.
  - Verify `st.session_state.optimization_complete` is true after solve.
- [ ] **Task 6: Implement Error State Test**
  - Test app behavior when solving with inconsistent data.
  - Verify error message display.

## Phase 3: Verification & Cleanup

- [ ] **Task 7: Manual Theme Validation**
  - Check Light and Dark mode appearance in a real browser.
  - Ensure "premium" aesthetics are preserved.
- [ ] **Task 8: Run Full Test Suite**
  - Run `pytest` to verify all co-located tests and the new integrated app tests.
- [ ] **Task 9: Final Documentation Sync**
  - Ensure all specs and plans are updated with implementation details.
