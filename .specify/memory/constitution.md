# Hybrid Quantamental Optimizer Constitution

## Core Principles

### I. Simplicity Through Deep Modules
The codebase MUST reduce complexity rather than relocate it. In the spirit of
John Ousterhout, each module MUST expose a small, stable interface and hide
operational detail behind it.

- New abstractions MUST make the system easier to understand, test, or change.
- Shallow wrappers, speculative layers, and wide configuration surfaces are
  prohibited.
- Complexity SHOULD be absorbed inside modules, not pushed onto callers.

### II. Small, Atomic Changes
Each change MUST represent one logical unit of value.

- Pull requests SHOULD stay as small as practical.
- A change that cannot be explained clearly in one sentence MUST be split.
- Review, rollback, and verification MUST remain simple.

### III. Verification Before Completion
No change is complete until it has passed a reproducible verification step.

- Verification MAY be tests, linting, type checks, builds, CLI runs, or other
  repeatable checks appropriate to the change.
- Output inspection alone is insufficient.
- Verification evidence MUST be recorded in the relevant plan, task list,
  pull request, or delivery summary.

### IV. Professional Clarity
Code and documentation MUST be concise, precise, and current.

- Names MUST communicate intent.
- Comments MUST add non-obvious information.
- Specifications MUST describe behavior and scope clearly.
- Redundant, vague, or low-signal documentation is prohibited.

### V. High-Leverage Delivery
Work MUST prioritize the smallest high-value slice that solves the current
problem.

- Prefer incremental delivery over speculative design.
- Default to the simplest solution that satisfies current requirements.
- Premature generalization is prohibited unless it removes more complexity than
  it introduces.

## Workflow Rules

- Feature specifications MUST stay focused on user-visible behavior, scope, and
  success criteria rather than implementation detail.
- Implementation plans MUST include a constitution check that confirms
  interface simplicity, scope discipline, and verification strategy.
- Task lists MUST include explicit verification work and keep user stories
  independently implementable where practical.
- UI-facing changes MUST be approached with the judgment of experienced UX and
  UI designers and researchers, including clear user flows, meaningful states,
  accessibility, and visual consistency.
- Multi-step work SHOULD be captured in meaningful checkpoint commits and
  synced at sensible milestones to reduce drift and review risk.
- Complexity added for a real need MUST be justified in the plan, including why
  a simpler approach was rejected.

## Boundaries

This constitution governs enduring engineering practice only.

The following MUST live outside the constitution:

- product scope
- technology choices
- UI direction
- provider or integration decisions
- feature-specific architecture

Those decisions belong in specifications, plans, ADRs, or operational
documentation.

## Governance

This constitution overrides conflicting local habits and informal process.

- Every review MUST confirm that interfaces remain simple and that complexity
  is hidden inside deep modules.
- Every review MUST confirm that scope is no larger than necessary for one
  logical change.
- Every review MUST confirm that verification was executed and documented.
- Amendments MUST be documented, reviewed for template impact, and versioned.

Versioning policy:

- MAJOR: removes or materially redefines a governing principle
- MINOR: adds a principle or materially expands governance
- PATCH: clarifies wording without changing intent

**Version**: 1.0.0 | **Ratified**: 2026-05-03 | **Last Amended**: 2026-05-03
