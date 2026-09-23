# Repository instructions

## Python changes

- Format Python sources and stubs with `ruff format python`.
- Run `ruff check python` and `pyright` before committing Python changes.
- Run `python -m pytest -q python/tests` against a freshly built extension wheel.
- Keep runtime exports and `python/sketch_spgemm/_sketch_spgemm.pyi` synchronized.
- Supported matrix value dtypes are `int32`, `int64`, `float32`, and `float64`.
  CSR indices and row pointers remain `int64`.
- `auto_spgemm` and `analyze_workload` are `int64`-only because the sketch
  pipeline uses exact signed 64-bit arithmetic. Use `checked_spgemm` for every
  supported dtype.
