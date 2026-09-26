# AGENTS.md

## General Instructions

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
- Keep comments focused on intent and non-obvious decisions rather than restating code.

## Repository instructions for coding agents

- After each completed change to this repository, create a git commit containing that change before reporting completion. If a request involves multiple related edits, commit them together once they have been verified.
- Before committing, inspect `git status`, `git diff`, and recent commits. Stage only files belonging to the change; leave unrelated or pre-existing work untouched.
- Run checks appropriate to the change and use a concise commit message consistent with repository history. If checks or a commit fail, report the blocker rather than claiming the change is complete.
