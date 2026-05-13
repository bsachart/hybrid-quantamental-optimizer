# Tasks: Plotly Dash UI Migration

## Infrastructure & Setup
- [ ] **T001** Install `dash` and `dash-bootstrap-components` dependencies.
- [ ] **T002** Create `src/assets/style.css` and port the "Premium Quant" design system from `streamlit_app.py`.
- [ ] **T003** Initialize `src/app.py` with a skeleton Dash layout and basic multi-column structure.

## Core UI Components
- [ ] **T004** Implement File Upload components with status indicators and sample CSV previews.
- [ ] **T005** Implement assumption controls (Sliders) for Lending/Borrowing rates and Target Volatility.
- [ ] **T006** Implement the primary "Solve Risky Portfolio" button with a loading spinner/state.

## Results & Visualization
- [ ] **T007** Implement `dcc.Store` callbacks to manage Tangency and Final Portfolio state.
- [ ] **T008** Replicate the Capital Market Line visualization in Plotly, including lending/borrowing segments and tooltips.
- [ ] **T009** Replicate the Allocation Bar Chart in Plotly.
- [ ] **T010** Implement the Allocation Table component using custom styled HTML table rows (porting the `.allocation-table` CSS).
- [ ] **T011** Implement metrics cards for key performance indicators (Return, Vol, Sharpe, Cash Weight).

## Testing & Validation
- [ ] **T012** Add integration tests in `src/app_test.py` using `dash.testing` to verify the end-to-end solve workflow.
- [ ] **T013** Verify responsive behavior and theme consistency across both Light and Dark modes (simulated via CSS classes if necessary).

## Cleanup & Documentation
- [ ] **T014** Archive Spec 003 (Streamlit theme/testing) as it is superseded by the Dash migration.
- [ ] **T015** Add a README note or script to run the new Dash app (`python src/app.py`).
