# Feature Specification: Single-Column Layout And Grouped Rate Controls

**Feature Branch**: `003-single-column-layout`  
**Created**: 2026-05-05  
**Status**: Draft  
**Input**: User description: "Avoid two columns whenever possible. Merge 'Final allocation' with 'Final allocation breakdown'. Split between borrowing and lending rates and move the three selectors (target volatility, lending and borrowing rates) closer to each other. Lending and borrowing rates can be on the same line."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read Results In One Unbroken Column (Priority: P1)

As an investor reviewing a solved portfolio, I want the results section to
flow in a single reading column so that my eye never has to jump sideways to
find the next piece of information.

**Why this priority**: Two-column layouts force the reader to manage two
competing reading streams simultaneously. In a financial context where each
number depends on the previous one, that lateral jump introduces exactly the
kind of friction that produces misreading and lost context. A single vertical
column enforces a clear causal order: assumption → result → implication →
action.

**Independent Test**: Solve the portfolio with the example files and scroll
top-to-bottom through the results section in a single pass, confirming every
meaningful element is reachable without horizontal scanning or side-by-side
comparison.

**Acceptance Scenarios**:

1. **Given** a solved portfolio is displayed, **When** the user reads from the
   tangency metrics down through the final allocation, **Then** every step in
   the results flow appears in a single vertical column and no content is
   placed side-by-side in a way that requires parallel reading.
2. **Given** the setup area is displayed, **When** the user scans the upload
   and solve controls, **Then** the controls appear in a single reading order
   and do not require the user to scan both a left and a right panel to
   understand what is needed.

---

### User Story 2 - See The Allocation Once, Completely (Priority: P2)

As an investor inspecting the final portfolio, I want the allocation bar chart
and the detailed allocation table to appear as one unified section so that I
understand the full breakdown in a single glance rather than discovering a
table below a separate chart section.

**Why this priority**: Separating the bar chart and the breakdown table
creates an invisible seam: the user sees a partial picture, draws a
conclusion, then finds more detail below that may correct their first
impression. Merging them into one section respects the user's attention and
removes the double-take. The chart provides immediate visual proportion; the
table provides precise figures — they answer the same question and belong
together.

**Independent Test**: Solve the portfolio and inspect the allocation section.
Confirm the bar chart and the detailed numeric table appear within one
contiguous section with no unrelated content between them.

**Acceptance Scenarios**:

1. **Given** the final portfolio is displayed, **When** the user looks at the
   allocation area, **Then** the horizontal bar chart and the detailed
   per-asset table appear in the same section without a section break or
   separate heading between them.
2. **Given** the final portfolio is displayed with both risky assets and cash,
   **When** the user reads the allocation section, **Then** the bar chart and
   table show consistent values for every row and the user can cross-reference
   them without scrolling.

---

### User Story 3 - Adjust All Three CML Controls From One Spot (Priority: P2)

As an investor tuning the final portfolio, I want the target volatility
slider, the lending rate display, and the borrowing rate slider to sit close
together so that I can see how changing any one of them affects the Capital
Market Line without hunting for the other two.

**Why this priority**: Target volatility, lending rate, and borrowing rate are
the three variables that fully describe where a portfolio sits on the Capital
Market Line. Each one constrains or shifts the others. When these controls are
spread across the page the cause-and-effect relationship is invisible: the
user adjusts volatility, scrolls to find the borrowing rate, adjusts it, and
must mentally reconstruct the connection. Grouping them creates a control
cluster with immediate feedback — the hallmark of a high-quality analytical
tool.

**Independent Test**: Solve the portfolio and locate the target volatility
slider. Confirm the lending and borrowing rate controls are visible without
scrolling away from the target volatility slider on a standard laptop screen.

**Acceptance Scenarios**:

1. **Given** a solved portfolio is displayed, **When** the user looks at the
   target volatility control, **Then** the lending rate and borrowing rate
   controls are visible in the same nearby area without requiring the user to
   scroll.
2. **Given** the user adjusts the borrowing rate, **When** the Capital Market
   Line chart updates, **Then** the user can see both the control and the
   chart result without moving away from the control cluster.

---

### User Story 4 - Understand Lending And Borrowing As Distinct But Paired Concepts (Priority: P3)

As an investor setting the rate assumptions, I want the lending rate and the
borrowing rate to appear as two clearly labelled but visually paired controls
on the same row so that I understand they are related but separate values that
define the two segments of the Capital Market Line.

**Why this priority**: Displaying lending and borrowing rates in the same
box, or one after the other in a single column, blurs an important conceptual
distinction: lending is a read-only value derived from the risk-free rate
assumed earlier; borrowing is an adjustable assumption that bends the CML
above the tangency portfolio. Placing them side-by-side on one row signals
parity while the visual affordance of each (static display vs interactive
slider) communicates their different roles.

**Independent Test**: Display the rate controls and confirm the lending rate
and borrowing rate appear on the same row, with the lending rate clearly
marked as a display-only value and the borrowing rate as an interactive
control.

**Acceptance Scenarios**:

1. **Given** a solved portfolio is displayed, **When** the user looks at the
   rate controls, **Then** the lending rate and borrowing rate appear on the
   same horizontal row with their own labels, distinguishable by visual
   affordance.
2. **Given** the user reads the rate labels, **When** they look at the lending
   rate, **Then** it is clear that this value comes from the risk-free rate
   assumption set during portfolio solve and is not editable here.
3. **Given** the user interacts with the borrowing rate, **When** they adjust
   it, **Then** only the borrowing rate changes and the lending rate remains
   stable, confirming they are independent values.

### Edge Cases

- What happens when lending rate and borrowing rate are equal? The kinked CML
  becomes a straight line — the merged row should remain visually coherent.
- What happens when the screen is narrow enough that a side-by-side row for
  lending and borrowing rates would wrap or become unreadable? The layout must
  remain usable.
- What happens when there are many risky assets in the allocation? The merged
  chart-and-table section must remain scrollable without obscuring the chart.
- What happens when the user resizes the browser window? Single-column
  preference should hold at all standard desktop widths.
- What happens when only cash is in the final allocation? The merged section
  should degrade gracefully with a single row.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The app MUST avoid side-by-side two-column layouts in the
  results flow unless the content is a direct left-right comparison that adds
  meaning only when seen simultaneously.
- **FR-002**: The app MUST avoid side-by-side two-column layouts in the setup
  flow; the upload area and the solve controls MUST read in a single vertical
  order.
- **FR-003**: The app MUST present the final allocation bar chart and the
  per-asset allocation table within one contiguous section under a single
  heading, with no unrelated content between them.
- **FR-004**: The app MUST place the target volatility slider, the lending
  rate display, and the borrowing rate slider within one compact control
  cluster so that all three are visible together on a standard laptop screen
  without scrolling.
- **FR-005**: The app MUST display the lending rate and the borrowing rate on
  the same row, each with its own label.
- **FR-006**: The lending rate control MUST be presented as a read-only
  display value, visually distinct from the interactive borrowing rate slider.
- **FR-007**: The app MUST preserve all existing calculations, chart behavior,
  and export functionality while reorganizing the layout.
- **FR-008**: The merged allocation section MUST show consistent values
  between the bar chart and the detailed table.

### Key Entities *(include if feature involves data)*

- **Control Cluster**: The group of three controls — target volatility,
  lending rate, and borrowing rate — that together determine position on the
  Capital Market Line.
- **Allocation Section**: The unified view that combines the allocation bar
  chart and the per-asset allocation table under a single heading.
- **Read-Only Rate Display**: A non-interactive presentation of the lending
  rate, visually differentiated from sliders and other interactive controls.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user reviewing results can locate the allocation
  bar chart and the per-asset breakdown within one visible section in under
  10 seconds.
- **SC-002**: A user adjusting portfolio position on the CML can find and
  interact with target volatility, lending rate, and borrowing rate without
  scrolling between them on a standard laptop viewport.
- **SC-003**: On any standard desktop width, the results section reads in a
  single top-to-bottom column with no content requiring lateral eye movement
  to parse a logical sequence.
- **SC-004**: Users can distinguish the lending rate (read-only) from the
  borrowing rate (interactive) on first inspection without reading surrounding
  explanatory text.

## Assumptions

- The feature targets the existing browser-based optimizer experience.
- The existing calculation behavior, chart output, and CSV export remain
  unchanged.
- "Two columns" in FR-001 and FR-002 refers to side-by-side panels that
  split the logical reading sequence, not to visual grids used purely for
  alignment of a single conceptual unit (e.g., a row of metric cards).
- A "standard laptop screen" is assumed to be approximately 1280–1440 px
  wide; all layout decisions must hold at this range.
- The lending rate is always derived from the risk-free rate set during
  portfolio solve and cannot be changed in the results section.

## Out of Scope

- Changes to the Capital Market Line calculation or engine logic
- New chart types or chart configuration options
- Mobile or narrow-viewport responsive redesign beyond graceful degradation
- New data fields or export formats
- Changes to the risk model selector or file upload behavior
