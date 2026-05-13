# Feature Specification: Plotly Dash UI Migration

**Feature Branch**: `004-plotly-dash-ui`  
**Created**: 2026-05-13  
**Status**: Draft  
**Input**: User description: "I want to switch to plotly in python instead of streamlit. Streamlit is too rigid and not flexible for our use case. Simple efficient portfolio calculator. Keep the good things like styling, ... and let's migrate to Plotly"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Portfolio Optimization Workflow (Priority: P1)

As a quant analyst, I want to upload price and metric files and solve for the tangency portfolio in a single, responsive interface, so that I can quickly arrive at a baseline risky allocation.

**Why this priority**: This is the core functional requirement of the application. Without the ability to load data and solve, the app has no value.

**Independent Test**: Can be tested by uploading sample CSVs, clicking "Solve", and verifying that the tangency portfolio metrics (Return, Vol, Sharpe) are displayed correctly.

**Acceptance Scenarios**:

1. **Given** valid price and metric CSVs, **When** "Solve Risky Portfolio" is triggered, **Then** the system computes the tangency portfolio and displays its key performance indicators.
2. **Given** invalid or missing data, **When** a solve is attempted, **Then** a clear, non-technical error message is displayed to the user.

---

### User Story 2 - Capital Market Line Interaction (Priority: P2)

As a portfolio manager, I want to interactively scale my portfolio along the Capital Market Line by adjusting target volatility and financing rates, so that I can see the impact on my final allocation and expected return.

**Why this priority**: This differentiates the tool from a simple static calculator and allows for "what-if" analysis of leverage and cash positions.

**Independent Test**: Can be tested by moving the "Target Volatility" slider and observing the "Final Portfolio" metrics and charts update in real-time.

**Acceptance Scenarios**:

1. **Given** a solved tangency portfolio, **When** target volatility is adjusted below the tangency point, **Then** the app shows a "Lending" portfolio with a cash yield component.
2. **Given** target volatility is adjusted above the tangency point, **When** the borrowing rate is set, **Then** the app shows a "Borrowing" portfolio with the appropriate financing cost applied.

---

### User Story 3 - Premium Visual Analysis (Priority: P3)

As a user, I want the application to maintain its premium "quantamental" aesthetic (beige/paper theme) and provide interactive Plotly visualizations, so that the experience feels professional and polished.

**Why this priority**: Maintains the "wow factor" and design standards established in the project constitution.

**Independent Test**: Manual visual inspection to ensure the color palette, typography, and chart styling match the previous Streamlit implementation's high standards.

**Acceptance Scenarios**:

1. **Given** the app is running, **When** viewed on different screen sizes, **Then** the layout remains responsive and maintains its high-contrast, premium styling.
2. **Given** interactive charts, **When** hovered over or zoomed, **Then** the tooltips and axis labels remain readable and consistent with the theme.

## Edge Cases

- **Large Dataset Performance**: How does the Dash callback system handle very large price history files? (Should provide visual feedback during computation).
- **Network Latency**: Since Dash uses a server-client model, how does the UI handle delays in file uploads or solve times? (Implement loading states).
- **Invalid Rate Configurations**: What happens if the borrowing rate is set lower than the lending rate? (The system should ideally enforce a floor or warn the user).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a highly flexible web-based interface that supports custom layouts and styling beyond standard template constraints.
- **FR-002**: System MUST support CSV file uploads for Price History and Asset Metrics.
- **FR-003**: System MUST implement a multi-stage workflow (Setup -> Solve -> Results).
- **FR-004**: System MUST allow interactive adjustment of lending/borrowing rates and target volatility.
- **FR-005**: System MUST render interactive visual representations of the Capital Market Line and Efficient Frontier.
- **FR-006**: System MUST display a detailed allocation table for the final portfolio.
- **FR-007**: System MUST support a custom design system (typography, color palette) that matches a premium "quantamental" aesthetic.

### Key Entities *(include if feature involves data)*

- **Price History**: Time-series data of asset prices.
- **Asset Metrics**: Expected returns, volatilities, and constraints for each ticker.
- **Tangency Portfolio**: The portfolio with the maximum Sharpe ratio on the efficient frontier.
- **Final Allocation**: The specific mix of assets and cash/borrowing that meets the user's target risk.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete the full workflow (Upload -> Solve -> Tune) in under 60 seconds.
- **SC-002**: 100% functional parity with the existing Streamlit application.
- **SC-003**: UI components (sliders, buttons, charts) respond to state changes in under 300ms (excluding optimization solve time).
- **SC-004**: Visual style score (manual) matches or exceeds the "premium" feel of the Streamlit version.

## Assumptions

- **Plotly Dash**: Assumes Plotly Dash is the intended technology for "Plotly in Python" to provide the required UI flexibility.
- **Logic Preservation**: Assumes the existing `engine` logic (pandas-based optimization) is robust and can be reused without modification.
- **Theme Consistency**: Assumes the "Gelasio" and "Urbanist" fonts and the specific beige color palette are the desired branding.
- **Single Session**: Assumes the app is primarily for single-user local or private deployment (stateless or session-based).
