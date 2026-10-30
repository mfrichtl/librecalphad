# AGENTS.md

## General Instructions

- Prioritize work on the LC-steels database. Treat known errors in `mf-steel.tdb` as out of scope unless a task explicitly asks to address them.
- Do not investigate or add regression tests for results generated from `mf-steel.tdb` in `mf-steel_validation` unless explicitly requested. Continue to run and address failures in the existing test suite for code you change.
- For LC-steels assessments, fit experimental CPM data using `heat_capacity.py` and ESPEI-datasets before using ESPEI to fit the database. Do not directly adjust thermodynamic parameters unless explicitly requested.
- Keep generated fit plots, fitted parameters, and database outputs separate from source inputs. Review results before using them downstream, and do not commit generated results unless explicitly requested.
- Make focused, minimal changes that directly address the requested task.
- Preserve existing behavior unless the request explicitly requires changing it.
- Follow the existing project structure, naming conventions, formatting, and coding style.
- Do not modify generated files, dependency lock files, or unrelated files unless necessary.
- Do not introduce new dependencies unless they are required and justified.
- Prefer clear, maintainable code over clever or overly abstract implementations.
- Add or update tests when behavior changes, where an appropriate test suite exists.
- Run relevant formatting, linting, type-checking, and test commands when available.
- Report files changed, the behavior changed, and any validation performed.
- Ask for clarification when requirements are ambiguous or when a change could alter public behavior.
- Avoid exposing secrets, credentials, private paths, or other sensitive data in code, logs, tests, or documentation.

## Unary reference-state export

- Use the companion `librecalphad-refstates` repository (normally a sibling checkout) and its ESPEI `LCRefState` plugin for fitted unary Gibbs reference states. Follow `docs/unary-reference-state-export.md`; do not replace this architecture with direct unary `G` writes without explicit approval.
- Export is coordinated across two destinations: `LCRefstates.json` for the Gibbs terms assembled by `create_espei_custom_refstate_stable`, and the database for native `THETA`, `BMAGN`, `TC`/`NT`, and `GD` parameters via `upsert_db_param_from_models`. Neither destination alone is a complete export.
- Map the stable element model to the bare element key (for example `FE`), its stable-phase alias to `FE-BCC_A2: "GHSERFE"`, and other unary phase models to `ELEMENT-PHASE` keys. Do not copy the nested `_fitted_params.json` wholesale into the plugin registry or add the stable Gibbs expression twice.
- Treat `LCSERparams.json` as separate reference metadata (`phase`, `mass`, `H298`, `S298`), not as fit parameters or automatic offset corrections. Establish the energy/entropy reference convention before updating it.
- Review optimizer diagnostics before promoting results. Preserve fitting metadata for provenance, but use named thermodynamic coefficients rather than optimizer indices or cached `variable_values` to define exported expressions.
- After approved registry changes, generate in a fresh Python process. `reload(espei)` is insufficient. Also inspect the seed database: ESPEI preserves existing `GHSER` symbols, unary `G` parameters, phase definitions, and SER entries, which can prevent new reference data from taking effect.
- Do not assume native magnetic and two-state contributions match the fitter. Check fitted `p`, the installed magnetic formula, gas constants, melt masks, site-ratio normalization, and double counting. Phase-wide magnetic hints affect other endmembers; do not change them globally to accommodate one unary fit without approval.
- Validate fitted Cp against pycalphad Cp before and after XML round-trip, including branch boundaries, then validate H/S reference conventions separately. Keep Debye export safeguards; do not silently drop unsupported contributions.
- Treat companion registry edits and database outputs as separate, explicitly authorized assessment changes. Preserve pre-existing work in both repositories and commit only requested source/documentation changes in each repository.

## Python Guidelines

- Use idiomatic Python and keep compatibility with the project's configured Python version.
- Prefer type hints for new or modified public functions when consistent with nearby code.
- Use descriptive names and keep functions small and single-purpose.
- Raise specific exceptions with actionable error messages.
- Do not use mutable objects as default argument values.
- Avoid broad exception handling unless errors are intentionally being handled or re-raised with context.

## Testing Guidelines

- Create or update appropriate unit tests for each code change.
- Tests should cover the requested behavior and relevant error cases.
- Run the focused unit test or tests covering the change and confirm they pass before running the full test suite.
- Run the complete test suite after focused tests pass.
- Report the focused test command, full test-suite command, and their results.
- Tests should be deterministic and independent of network access, local machine configuration, and execution order.
- Do not weaken or remove existing tests merely to make a change pass.
- Use temporary directories and fixtures for filesystem interactions where possible.

## Documentation Guidelines

- Update documentation when public APIs, configuration, command-line behavior, or user-facing workflows change.
- Update `AGENTS.md` when new repository-specific guidance would be useful; keep it current as workflows and priorities evolve.
- Keep comments focused on intent and non-obvious decisions rather than restating code.

## Repository instructions for coding agents

- After each completed change to this repository, create a git commit containing that change before reporting completion. If a request involves multiple related edits, commit them together once they have been verified.
- Before committing, inspect `git status`, `git diff`, and recent commits. Stage only files belonging to the change; leave unrelated or pre-existing work untouched.
- Run checks appropriate to the change and use a concise commit message consistent with repository history. If checks or a commit fail, report the blocker rather than claiming the change is complete.
