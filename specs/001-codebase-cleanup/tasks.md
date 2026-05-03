# Tasks: Codebase Cleanup And Calculation Verification

**Input**: Design documents from `/specs/001-codebase-cleanup/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/python-api.md, quickstart.md

**Tests**: Deterministic verification is required for this feature. Write or extend tests before refactoring the targeted calculation path.

**Organization**: Tasks are grouped by user story so each slice can be implemented and verified independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (`US1`, `US2`, `US3`)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the verification baseline and confirm the active cleanup seam

- [ ] T001 Run the baseline engine verification command from `specs/001-codebase-cleanup/quickstart.md`
- [ ] T002 Confirm the preserved public behavior in `specs/001-codebase-cleanup/contracts/python-api.md` before refactoring

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create the shared helper and direct test surface that later story work depends on

- [ ] T003 [P] Create `src/engine/portfolio_math.py` for shared labeled-portfolio assembly, target-volatility scaling, and capital-market-line target generation
- [ ] T004 [P] Create `src/engine/portfolio_math_test.py` with deterministic unit coverage for shared portfolio math invariants

**Checkpoint**: Shared portfolio-math helper exists and has direct test coverage

---

## Phase 3: User Story 1 - Trust Core Portfolio Calculations (Priority: P1) 🎯 MVP

**Goal**: Preserve and verify the core optimizer calculation path during cleanup

**Independent Test**: Run deterministic engine tests that validate tangency construction, target-volatility scaling, cash allocation, and CML generation

### Tests for User Story 1

- [ ] T005 [P] [US1] Extend `src/engine/portfolio_engine_test.py` with exact capital-market-line and cash-allocation assertions for scalar and list target-volatility paths

### Implementation for User Story 1

- [ ] T006 [US1] Refactor `src/engine/portfolio_engine.py` to use `src/engine/portfolio_math.py` while preserving the public return structure
- [ ] T007 [US1] Run `pytest src/engine/portfolio_math_test.py src/engine/portfolio_engine_test.py src/engine/optimizer_test.py src/engine/risk_test.py`

**Checkpoint**: The active polars engine path remains correct and fully testable

---

## Phase 4: User Story 2 - Reduce Engine Complexity For Maintainers (Priority: P2)

**Goal**: Remove duplicated portfolio math from the secondary engine path without broad backend unification

**Independent Test**: Run deterministic shared-helper tests plus a pandas-engine smoke test that confirms labeled outputs and target-volatility scaling still work

### Tests for User Story 2

- [ ] T008 [P] [US2] Add `src/engine/portfolio_engine_pandas_test.py` with deterministic smoke coverage for optimization output shape and target-volatility scaling

### Implementation for User Story 2

- [ ] T009 [US2] Refactor `src/engine/portfolio_engine_pandas.py` to use `src/engine/portfolio_math.py` and remove duplicated scaling logic
- [ ] T010 [US2] Run `pytest src/engine/portfolio_engine_pandas_test.py src/engine/portfolio_math_test.py src/engine/portfolio_engine_test.py`

**Checkpoint**: Both engine entry points share the same portfolio-scaling logic

---

## Phase 5: User Story 3 - Keep The Repository Operationally Clean (Priority: P3)

**Goal**: Reduce avoidable repository noise while keeping workflow guidance explicit

**Independent Test**: Confirm generated cache artifacts are ignored or removed and inspect the working tree for only intentional changes

### Implementation for User Story 3

- [ ] T011 [P] [US3] Verify `.gitignore` coverage and remove local generated cache directories such as `src/**/__pycache__/` and `src/engine/.pytest_cache/` if present
- [ ] T012 [US3] Confirm `AGENTS.md` and `.specify/memory/constitution.md` reflect checkpoint commit and sync expectations without extra workflow noise
- [ ] T013 [US3] Run `git status --short` and confirm only intentional feature files remain

**Checkpoint**: Local repository noise is reduced and workflow guidance is explicit

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final clarity and full verification across the cleanup slice

- [ ] T014 [P] Review docstrings and comments in `src/engine/portfolio_engine.py`, `src/engine/portfolio_engine_pandas.py`, and `src/engine/portfolio_math.py` for concise clarity after refactoring
- [ ] T015 Run `pytest src/engine` and record final verification evidence for the cleanup slice

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately
- **Foundational (Phase 2)**: Depends on Setup completion
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion and should follow User Story 1 because it reuses the shared helper
- **User Story 3 (Phase 5)**: Can run after the implementation shape is mostly settled
- **Polish (Phase 6)**: Depends on the desired user stories being complete

### Within Each User Story

- Tests must be added or extended before the matching refactor
- Shared helper work must land before either engine path consumes it
- Verification must run before the task group is considered complete

### Parallel Opportunities

- `T003` and `T004` can proceed in parallel
- `T005` and `T008` touch different verification files and can be prepared independently
- `T011` and `T014` can run alongside late-stage verification once code changes stabilize

## Implementation Strategy

### MVP First

1. Complete setup and foundational work
2. Complete User Story 1
3. Validate the active polars engine path
4. Expand to the pandas engine path
5. Finish hygiene and polish

### Notes

- Keep this cleanup incremental
- Preserve the public portfolio API
- Prefer deep shared helpers over shallow wrappers
- Mark tasks complete in this file as work lands
