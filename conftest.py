"""What pytest never collects unless named: files that do their work on import."""

# pyproject.toml's --ignore is relative to the directory pytest runs in, and with
# --doctest-modules `pytest .` imports every .py under the root, scripts/ and tools/ too.
# This file is listed as well: collected as a doctest module, it would clash with
# tests/conftest.py, which is imported under the same name. An argument naming one of
# these (`pytest scripts/tour.py`) still collects it; the scripts refuse to be imported.
collect_ignore = ["conftest.py", "tests/reference", "scripts", "tools"]
