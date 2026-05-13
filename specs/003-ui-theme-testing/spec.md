# Feature Specification: UI Theme Support & Integrated Testing

**Feature Branch**: `003-ui-theme-testing`  
**Created**: 2026-05-10  
**Status**: Draft  
**Input**: User description: "Properly support light and dark mode. The current UI is broken. Add Streamlit unit tests."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Theme Responsiveness (Priority: P1)

As a user, I want the application to automatically adapt its visual theme (light or dark) based on my system settings or Streamlit theme preferences, so that the interface is readable and aesthetically pleasing in any environment.

**Why this priority**: High priority because the current UI is reported as "broken" (likely due to fixed light-mode colors in a dark-mode environment) and cross-theme support is a core modern UX requirement.

**Independent Test**: Can be tested by switching the system/browser theme or Streamlit theme setting and observing that background, panel, and text colors update correctly without hardcoded "beige" artifacts in dark mode.

**Acceptance Scenarios**:

1. **Given** a system in dark mode, **When** the app loads, **Then** all backgrounds, panels, and text colors are dark-themed with appropriate contrast.
2. **Given** a system in light mode, **When** the app loads, **Then** the premium "quant" aesthetics (beige/paper) are preserved with high readability.

---

### User Story 2 - Integrated App Verification (Priority: P2)

As a developer, I want to have automated tests that simulate the entire application workflow (file upload, solve, results display), so that I can verify that UI changes don't break the core functionality.

**Why this priority**: Essential for maintainability and "bug prevention" as per the project constitution. It moves beyond pure logic testing to application-state testing.

**Independent Test**: Can be tested by running `pytest` on the new Streamlit app tests and seeing them pass or fail based on actual widget interactions.

**Acceptance Scenarios**:

1. **Given** the app is running, **When** valid price and metric files are uploaded and "Solve" is clicked, **Then** the results section appears and contains correct summary metrics.
2. **Given** the app is running, **When** a solve fails due to invalid data, **Then** a user-friendly error card is displayed.

---

### User Story 3 - Visual Consistency & Polish (Priority: P3)

As a user, I want the UI elements (buttons, charts, tables) to have consistent and accessible contrast across all themes, so that I can easily navigate and interpret the portfolio results.

**Why this priority**: Completes the "premium" feel and addresses the "broken UI" feedback by ensuring charts and tables are not just "dark" but well-designed for dark mode.

**Independent Test**: Manual visual inspection of chart legends, axis labels, and table headers in both themes.

**Acceptance Scenarios**:

1. **Given** dark mode is active, **When** a chart is rendered, **Then** the grid lines, axis labels, and legends use accessible light colors.
2. **Given** light mode is active, **When** a table is rendered, **Then** the header and row colors follow the established design system.

## Edge Cases

- **Mixed Theme State**: What happens if the Streamlit theme is set to "Light" but the system is "Dark"? (System should ideally follow Streamlit's internal variable state if available, or fall back gracefully).
- **Missing Data during Test**: How does the `AppTest` handle scenarios where files are only partially uploaded? (It should confirm the "Solve" button remains disabled).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define CSS variables for all colors (background, panel, ink, muted, accent) that respond to `prefers-color-scheme: dark` or Streamlit's built-in theme variables.
- **FR-002**: Charts (Altair) MUST have theme-aware background and axis configurations.
- **FR-003**: The "Solve" button MUST be tested via automated Streamlit `AppTest` suite.
- **FR-004**: The UI MUST maintain the "premium quant" aesthetic in both modes, avoiding default/generic "black and white" styles.
- **FR-005**: All hardcoded hex codes in `streamlit_app.py` and `results_display.py` MUST be replaced with CSS variables or dynamic theme-aware values.

### Key Entities *(include if feature involves data)*

- **App Theme State**: Represents the current visual mode (Light/Dark).
- **App Test Runner**: Represents the automated test environment for the Streamlit application.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of hardcoded "light-only" colors in CSS are converted to theme-aware variables.
- **SC-002**: At least 3 automated integration tests cover the "Happy Path" (Upload -> Solve -> Results).
- **SC-003**: UI contrast ratios for primary text meet WCAG AA standards in both Light and Dark modes.
- **SC-004**: Zero "broken" visual artifacts (e.g., white-on-white text or dark-on-dark buttons) reported in manual verification of both modes.

## Assumptions

- **Streamlit Version**: Assumes Streamlit 1.28.0+ is available for `AppTest` (verified 1.52.1).
- **CSS Injection**: Assumes the current method of `st.markdown("<style>...", unsafe_allow_html=True)` is the preferred way to maintain the custom design system.
- **Design Intent**: Assumes the "paper/beige" look is the desired Light mode, and a "deep slate/charcoal" look is the desired Dark mode.
