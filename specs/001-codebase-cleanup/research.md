# Research: Codebase Cleanup And Calculation Verification

## Decision: Verify calculation invariants before refactoring

- **Rationale**: Cleanup work on portfolio construction is only safe when
  tangency construction, volatility targeting, and covariance behavior are
  protected by deterministic checks.
- **Alternatives considered**:
  - Refactor first and rely on existing broad tests. Rejected because the
    duplicated math paths are exactly where silent drift can hide.
  - Use only manual output inspection. Rejected because it does not satisfy the
    constitution's verification requirement.

## Decision: Target shared portfolio math as the first cleanup seam

- **Rationale**: `portfolio_engine.py` and `portfolio_engine_pandas.py` duplicate
  the same post-optimization calculations, capital-market-line scaling, and cash
  portfolio construction. Extracting that logic into a shared helper reduces
  complexity while preserving both data-loading backends.
- **Alternatives considered**:
  - Unify all pandas and polars loaders immediately. Rejected because the loader
    paths have meaningful behavioral differences and would expand scope.
  - Clean only comments and formatting. Rejected because it would not materially
    reduce calculation complexity.

## Decision: Preserve separate loader and risk backends for this slice

- **Rationale**: The repository uses distinct polars and pandas paths for
  different runtime environments. A shared portfolio-math layer gives leverage
  without forcing immediate backend consolidation.
- **Alternatives considered**:
  - Remove the pandas path now. Rejected because it appears tied to the browser
    deployment path.
  - Build a full abstraction layer over all dataframe operations. Rejected as a
    shallow layer with low immediate leverage.

## Decision: Treat repository hygiene as lightweight supporting work

- **Rationale**: Generated caches and similar artifacts add noise, but they are
  supporting cleanup rather than the center of the feature. Hygiene work should
  stay small and avoid distracting from calculation confidence.
- **Alternatives considered**:
  - Broad repository restructuring. Rejected because it is outside the first
    incremental cleanup scope.
