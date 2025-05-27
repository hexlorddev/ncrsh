# Contributing to ncrsh

Thank you for your interest in contributing to ncrsh! We welcome contributions from the community to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as [GitHub issues](https://github.com/dinethnethsara/ncrsh/issues).

When you are creating a bug report, please include as many details as possible:

- A clear and descriptive title
- A description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/dinethnethsara/ncrsh/issues).

When you are creating an enhancement suggestion, please include:

- A clear and descriptive title
- A description of the suggested enhancement
- Why this enhancement would be useful
- Examples of how it might be used

### Pull Requests

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ncrsh.git
   cd ncrsh
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for static type checking
- **flake8** for linting

Before committing, please run:

```bash
black .
isort .
flake8
mypy .
```

## Testing

We use `pytest` for testing. To run the tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=ncrsh tests/
```

## Documentation

We use Sphinx for documentation. To build the documentation:

```bash
cd docs
make html
```

The documentation will be available in `docs/_build/html`.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
