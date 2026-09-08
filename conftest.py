# Empty root conftest.py so pytest adds the repository root to sys.path.
#
# tests/test_pipeline.py imports `analysis` and `load_data`, which live in
# the repo root. tests/ has no __init__.py, so pytest's default "prepend"
# import mode only adds tests/ itself to sys.path when collecting test
# modules -- not the repo root. A root-level conftest.py (even empty) is
# always imported with its directory prepended to sys.path, which makes
# the root-level modules importable during test collection without
# requiring PYTHONPATH to be set manually before running `pytest`/`make test`.
