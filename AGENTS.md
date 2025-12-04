## Build, Lint, and Test

- **Install dependencies**: `pip install -r requirements.txt`
- **Run all tests**: `pytest`
- **Run a single test file**: `pytest path/to/test_file.py`
- **Run a single test function**: `pytest path/to/test_file.py::test_function_name`
- **Linting**: No linter is configured. Please adhere to the code style guidelines below.

## Code Style Guidelines

- **Imports**: Place all imports at the top of the file. Use standard import aliases (e.g., `import numpy as np`).
- **Formatting**:
  - Indentation: 4 spaces.
  - Line length: Keep lines under 90 characters.
  - Docstrings: Use numpy-style docstrings.
- **Types**: Use type hints in function signatures.
- **Naming Conventions**:
  - Functions and variables: `snake_case`.
  - Constants: `UPPER_CASE`.
- **Error Handling**: Use `try...except` blocks for code that might raise exceptions.
- **General**: Follow existing code patterns and conventions.
