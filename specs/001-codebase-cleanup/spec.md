# Feature Specification: Codebase Cleanup And Calculation Verification

**Feature Branch**: `001-codebase-cleanup`  
**Created**: 2026-05-03  
**Status**: Draft  
**Input**: User description: "Clean up the codebase, reduce complexity, verify calculations, and work through specify -> plan -> tasks -> implement."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Trust Core Portfolio Calculations (Priority: P1)

As a user of the optimizer API, I want the portfolio calculations to be
verified against deterministic scenarios so that expected return, volatility,
cash scaling, and Sharpe outputs remain trustworthy during refactors.

**Why this priority**: Correct portfolio outputs are the core value of the
project. Cleanup without calculation confidence would increase risk.

**Independent Test**: This can be fully tested by running the engine test suite
against fixed price and metric inputs and verifying known outputs for tangency
portfolio construction, target-volatility scaling, and covariance behavior.

**Acceptance Scenarios**:

1. **Given** valid price and metric inputs, **When** the optimizer builds a
   tangency portfolio, **Then** the reported weights, expected return,
   volatility, and Sharpe ratio are internally consistent.
2. **Given** a tangency portfolio and a lower target volatility, **When** a
   scaled portfolio is generated, **Then** the equity weights and cash weight
   preserve the expected capital-market-line relationship.
3. **Given** a valid forward-looking volatility vector, **When** a covariance
   matrix is generated, **Then** the diagonal and off-diagonal values follow
   the documented risk model.

---

### User Story 2 - Reduce Engine Complexity For Maintainers (Priority: P2)

As a maintainer, I want core engine modules to have clearer responsibilities and
less duplicated logic so that future changes are easier to reason about and
verify.

**Why this priority**: The codebase already contains overlapping calculation and
validation paths. Simpler deep modules lower maintenance cost and reduce the
chance of silent divergence.

**Independent Test**: This can be tested independently by reviewing the changed
module boundaries and running the relevant tests for data loading, risk, and
portfolio assembly without requiring UI or deployment work.

**Acceptance Scenarios**:

1. **Given** the engine modules before cleanup, **When** the refactor is
   complete, **Then** shared calculation and validation paths are easier to
   trace and duplicated logic is reduced in the targeted files.
2. **Given** invalid input data, **When** the data loading path runs, **Then**
   validation failures remain explicit and consistent after refactoring.

---

### User Story 3 - Keep The Repository Operationally Clean (Priority: P3)

As a maintainer, I want generated artifacts and lightweight workflow rules kept
under control so that day-to-day work stays focused and reviewable.

**Why this priority**: Repository noise increases cognitive load and makes
incremental cleanup harder to review.

**Independent Test**: This can be tested independently by confirming generated
artifacts are ignored or removed as appropriate and by checking that the
working guidance captures checkpoint commit and sync expectations.

**Acceptance Scenarios**:

1. **Given** the repository contains generated caches or similar noise,
   **When** hygiene cleanup is complete, **Then** tracked or newly generated
   artifacts no longer create avoidable churn.
2. **Given** multi-step cleanup work, **When** contributors read project
   guidance, **Then** they can see the expectation to use meaningful checkpoint
   commits and sync at sensible milestones.

### Edge Cases

- What happens when the optimizer receives a zero-volatility tangency portfolio?
- What happens when the requested target volatility exceeds the tangency
  portfolio volatility?
- How does the system behave when price and metrics files only partially overlap
  by ticker?
- How does the system handle invalid or non-positive implied volatility values?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST preserve correct portfolio outputs for tangency
  portfolio construction after cleanup work.
- **FR-002**: The system MUST preserve correct target-volatility scaling,
  including cash allocation behavior for low-volatility and zero-volatility
  cases.
- **FR-003**: The system MUST preserve correct covariance construction for both
  supported risk-model paths covered by the current engine.
- **FR-004**: The system MUST keep input validation explicit for required price,
  metric, and weight-bound fields used by optimization.
- **FR-005**: The cleanup MUST reduce avoidable complexity in the targeted
  engine modules without changing documented external behavior.
- **FR-006**: The repository MUST avoid unnecessary churn from generated caches
  and similar non-source artifacts involved in local development.
- **FR-007**: Project guidance MUST state that meaningful checkpoint commits and
  sensible sync milestones are expected during multi-step work.

### Key Entities *(include if feature involves data)*

- **Universe**: The aligned price-history and asset-metrics dataset used as the
  input to risk modeling and optimization.
- **Portfolio Metrics**: The resulting portfolio weights, expected return,
  volatility, Sharpe ratio, and cash weight returned by the engine.
- **Risk Model**: The selected covariance construction mode that defines how
  price history and forward-looking volatility inputs are interpreted.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Deterministic verification scenarios cover tangency portfolio
  construction, volatility targeting, and covariance construction for the
  targeted cleanup scope.
- **SC-002**: The cleaned-up engine path produces the same expected return,
  volatility, and cash-allocation outcomes as the pre-cleanup behavior for the
  retained scenarios.
- **SC-003**: Invalid input scenarios for required metrics and weight bounds
  still fail with explicit validation outcomes after refactoring.
- **SC-004**: Repository hygiene changes eliminate avoidable development churn
  from generated artifacts in the targeted cleanup scope.

## Assumptions

- This feature targets code cleanup and verification rather than new optimizer
  capabilities.
- The current public Python API remains the compatibility boundary for this
  cleanup pass.
- The primary cleanup focus is the active engine and repository paths already in
  use by the existing tests.
- Large architectural replacement of the full pandas and polars split is out of
  scope for this first incremental pass unless a small shared extraction clearly
  reduces complexity.

## Out of Scope

- New portfolio construction features or new risk models
- UI redesign or product-scope changes
- Large repository restructuring unrelated to calculation confidence or cleanup
- Broad performance work not required to preserve correctness and simplicity
