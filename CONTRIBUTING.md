# Contributing to NSCK

Thank you for your interest in contributing to the Neural-Symbolic Cognitive Kernel (NSCK)! Contributions of all kinds are welcome — bug reports, feature suggestions, documentation improvements, and code patches.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request a Feature](#how-to-request-a-feature)
- [Development Setup](#development-setup)
- [Making a Pull Request](#making-a-pull-request)
- [Coding Style](#coding-style)
- [Running the Tests](#running-the-tests)
- [Good First Issues](#good-first-issues)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms.

---

## How to Report a Bug

1. **Search existing issues** first — the bug may already be reported.
2. Open a new issue using the **Bug Report** template.
3. Include:
   - A clear, descriptive title.
   - Steps to reproduce the problem.
   - Expected vs. actual behaviour.
   - Python version, OS, and any relevant dependency versions (`pip freeze`).
   - A minimal code snippet or stack trace if applicable.

---

## How to Request a Feature

1. **Search existing issues** to avoid duplicates.
2. Open a new issue using the **Feature Request** template.
3. Describe the problem your feature would solve and your proposed solution.

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/Node_network.git
cd Node_network

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Build the Rust accelerator for 21–206× speedup
cd nsck/rust_vsa && pip install -e . && cd ../..
cd nsck/rust_snn && pip install -e . && cd ../..

# 5. Run the test suite to confirm everything works
python -m pytest nsck/tests/ nsck_ai_model/tests/ -q
```

---

## Making a Pull Request

1. **Create a branch** off `main` with a descriptive name:
   ```bash
   git checkout -b fix/describe-the-fix
   git checkout -b feat/describe-the-feature
   ```
2. Make your changes following the [Coding Style](#coding-style) guidelines.
3. Add or update tests as appropriate.
4. Make sure the full test suite passes:
   ```bash
   python -m pytest nsck/tests/ nsck_ai_model/tests/ -q
   ```
5. Push your branch and open a Pull Request against `main`.
6. Fill in the PR template, link any related issues, and describe your changes clearly.

PRs that add tests for new behaviour, update documentation, and pass CI will be reviewed and merged faster.

---

## Coding Style

- Follow **PEP 8** for Python code.
- Use **type hints** for all public function signatures.
- Keep lines to **100 characters** or fewer.
- Write docstrings for all public classes and functions.
- Prefer descriptive variable names over terse abbreviations.
- New subsystems should match the patterns in existing modules (e.g. `nsck/python/core/`).

---

## Running the Tests

```bash
# Run all tests quietly
python -m pytest nsck/tests/ nsck_ai_model/tests/ -q

# Run a specific test file
python -m pytest nsck/tests/test_cognitive_engine.py -v

# Run with coverage (requires pytest-cov)
python -m pytest nsck/tests/ --cov=nsck --cov-report=term-missing -q
```

---

## Good First Issues

Look for issues labelled **`good first issue`** — these are deliberately scoped to be approachable for new contributors.

---

Thank you for helping make NSCK better! 🚀
