# GitHub Copilot Instructions

## Priorities (in order)
1) Correctness and security
2) Maintainability (clear structure, small functions, tests)
3) Performance where it matters (avoid obvious inefficiencies)
4) Consistency with existing repository style and patterns

## General behavior
- Follow the existing codebase conventions first (naming, layout, patterns, error handling).
- Do not introduce new abstractions unless they reduce complexity or duplication.
- Prefer minimal, incremental changes over large refactors.
- If requirements are ambiguous, implement the safest, most conservative behavior and document assumptions in code comments.

## Python (default rules when this is a Python repo)
- Target the repo’s configured Python version; if unknown, assume Python 3.12.
- Use ruff-compatible PEP 8 formatting. Line length: 120.
- Use double quotes for strings and f-strings for interpolation.
- Use built-in generics (list[str], dict[str, int]) and avoid importing from `typing` unless necessary.
- 100% type coverage: annotate all functions and public class members.
- Use Google-style docstrings for public APIs. Do not repeat type hints already in the signature.

## Naming
- Use descriptive names; avoid unclear abbreviations.
- Follow the repo’s naming convention. If the repo has no clear convention:
  - functions/variables: snake_case
  - classes: PascalCase nouns
  - constants: UPPER_SNAKE
  - booleans: is_/has_/can_/should_/did_

## Error handling
- Validate inputs early and fail with actionable error messages.
- Raise specific exceptions (custom types if the repo already uses them).
- Do not swallow exceptions silently; either handle them with a clear fallback or re-raise with context.

## Security rules (always)
- Never use `shell=True` with subprocess.
- Never use `eval()` or `exec()` on untrusted input.
- Avoid unsafe deserialization (e.g., `pickle`) unless explicitly required and documented.
- For YAML, use `yaml.safe_load()`.

## Dependencies
- Avoid adding new dependencies unless strictly necessary.
- If adding a dependency is unavoidable, explain why in a short comment near the import or in the PR description.
- If adding a dependency is unavoidable, make sure to update the LICENSE and THIRD_PARTY files.

## Performance
- Avoid obvious inefficiencies (e.g., quadratic loops, repeated heavy work in hot paths).
- Prefer vectorized/library operations over manual loops when processing large data.
- Avoid unnecessary data copies; be explicit about conversions.

## Testing (pytest or repo default)
- Update or add tests for behavior changes.
- Include: happy path, edge case, and failure case for public functions.
- Mock external I/O (filesystem/network/subprocess). Do not mock pure logic.

## Documentation
- Update the architecture documentation (docs folder) and the project README.md if you make changes that affect the overall design or framework logic.

## Output rules
- Provide complete implementations (no truncation).
- If something must remain unimplemented, mark it with a clear `TODO:` comment explaining what’s missing and why.