# Tasks: Plotly Dash UI Migration

## Infrastructure & Setup
- [x] **T001** Install `dash` and `dash-bootstrap-components` dependencies.
- [x] **T002** Create `src/assets/style.css` and port the "Premium Quant" design system from `streamlit_app.py`.
- [x] **T003** Initialize `src/app.py` with a skeleton Dash layout and basic multi-column structure.

## Core UI Components
- [x] **T004** Implement File Upload components with status indicators and sample CSV previews.
- [x] **T005** Implement assumption controls (Sliders) for Lending/Borrowing rates and Target Volatility.
- [x] **T006** Implement the primary "Solve Risky Portfolio" button with a loading spinner/state.

## Results & Visualization
- [x] **T007** Implement `dcc.Store` callbacks to manage Tangency and Final Portfolio state.
- [x] **T008** Replicate the Capital Market Line visualization in Plotly, including lending/borrowing segments and tooltips.
- [x] **T009** Replicate the Allocation Bar Chart in Plotly.
- [x] **T010** Implement the Allocation Table component using custom styled HTML table rows (porting the `.allocation-table` CSS).
- [x] **T011** Implement metrics cards for key performance indicators (Return, Vol, Sharpe, Cash Weight).

## Testing & Validation
- [x] **T012** Add integration tests in `src/app_test.py` using `dash.testing` to verify the end-to-end solve workflow.
- [x] **T013** Verify responsive behavior and theme consistency across both Light and Dark modes (simulated via CSS classes if necessary).

## Cleanup & Documentation
- [x] **T014** Archive Spec 003 (Streamlit theme/testing) as it is superseded by the Dash migration.
- [x] **T015** Add a README note or script to run the new Dash app (`python src/app.py`).
