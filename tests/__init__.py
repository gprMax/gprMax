# tests/__init__.py
#
# Makes the 'tests' directory a proper Python package.
#
# Why this is needed for CI:
#   pytest's default import mode (prepend) adds the repo root to sys.path
#   when it discovers test files.  In some edge cases with editable installs
#   (pip install -e .) and complex package structures, making tests/ a package
#   prevents import shadowing between gprMax modules and test modules that may
#   share a name.
#
#   Using setup.cfg's [tool:pytest] import-mode = importlib (set there) is the
#   more modern solution, but having __init__.py present ensures backward
#   compatibility with older pytest versions used by contributors.
