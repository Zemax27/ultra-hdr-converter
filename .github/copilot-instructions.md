# GitHub Copilot Instructions for Ultra HDR Converter

## Coding Conventions

### Python Instructions
- Ensure functions have descriptive names (avoid terse names like "i", "c") and include clear, concise comments.
- Use the `typing` module for type annotations (e.g., `list[str]`, `dict[str, int]`).
- Provide docstrings following PEP 257 / Google conventions. **Do not repeat type hints in the docstring.**
- Break down complex functions into smaller, more manageable functions.
- Prefer package imports rooted at `ultra_hdr_converter` in application-facing code.
- Keep numeric processing deterministic and explicit about dtypes (`np.float32` for linear-light math and `np.uint8` for encoded gain maps).

### General Guidelines
- Prioritize readability and clarity. Write concise, efficient, and idiomatic code.
- For algorithm-related code, include explanations of the approach used and comment on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- Avoid adding new libraries; only introduce dependencies when strictly necessary and justified. Mention external dependency usage and purpose in comments.
- Update the architecture documentation (docs folder) and the project README.md if you make changes that affect the overall design or framework logic.

### Code Style and Formatting
- Follow PEP 8 for Python code style.
- Always use **ruff** for linting and formatting Python code. Follow configuration rules:
  - Target Python version: 3.12
  - Line length: 120
  - Docstrings: Google convention (pydocstyle)
  - Enabled rule families: E, F, PL, I, N, A
  - Ignored rules: E501, PLR0913

### Workflow Expectations

- Validate gain map shape/channels against SDR base dimensions before encoding.
- Keep CLI arguments backward compatible when possible.
- Add or update tests in `tests/` for behavioral changes.

### Example of Proper Documentation and Type Hints
```python
def calculate_area(radius: float) -> float:
    """Calculate the area of a circle given the radius.

    Args:
        radius: The radius of the circle.

    Returns:
        The area of the circle, calculated as π * radius^2.

    Raises:
        ValueError: If the radius is negative.
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative.")

    return math.pi * radius ** 2
```