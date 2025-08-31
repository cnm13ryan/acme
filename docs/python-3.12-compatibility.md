---
title: Python 3.12 Compatibility
---

# Python 3.12 Compatibility and Dependency Guidance

This project targets Python 3.12 as the baseline interpreter.

- Required version: Python >= 3.12 (declared in `pyproject.toml`) and `.python-version` pins `3.12` for local tooling.
- Package metadata is defined via PEP 621 in `pyproject.toml`; the legacy `setup.py` remains for historical context and should not be used as the canonical source.

## Quick Start (3.12)

Choose one of the following flows.

### Using `uv` (recommended)

```
uv venv -p 3.12
source .venv/bin/activate
uv pip install -e .[test]
uv pip check
pytest -q acme
```

### Using `pip`

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .[test]
pip check
pytest -q acme
```

## What’s 3.12‑ready

- Core runtime deps: `absl-py`, `dm-env`, `dm-tree`, `numpy`, `pillow`, `typing-extensions`.
  - Note: Some platforms may require prebuilt wheels (e.g., `dm-tree`). If a wheel is unavailable for your OS/arch, a local build toolchain is needed.

## Known Rough Edges (Extras)

These extras are optional and may require adjustments on Python 3.12. If you don’t need them, skip installing the extra to avoid resolver friction.

- `testing`: includes `pytype>=2024.04.11` with Python 3.12 support (updated from legacy version).
  - Also includes `pytest>=8.0.0` with modern compatibility.

- `notebook`: `ipykernel==6.17.1` is several years old. Prefer newer `ipykernel` on 3.12 if you need notebooks.

- `envs` (simulated environments): includes `dm-control`, `gymnasium==1.2.0`, `gymnasium[atari]`, `pygame==2.6.1`.
  - Updated from deprecated `gym` to `gymnasium` for Python 3.12 compatibility.
  - **Removed incompatible packages**: 
    - `atari-py`: Build issues on Python 3.12
    - `bsuite`: Uses deprecated `imp` module (not Python 3.12 compatible)  
    - `rlds`: Only supports Python 3.7-3.10
  - Modern alternatives available as separate installs:
    - Use `stable-baselines3` for comprehensive RL algorithms
    - Use `gymnasium[atari]` for Atari environments (replaces `atari-py`)
    - Install `rlds` separately only on Python <3.11 for offline RL datasets

- `jax` stack: `jax>=0.4.35` (latest: 0.7.1), `jaxlib>=0.4.35`, `flax>=0.9.0`, `optax>=0.2.3`, `chex>=0.1.89`.
  - All dependencies now support Python 3.12.
  - Ensure a `jaxlib` wheel exists for your OS/arch/Python (CPU/GPU). Follow the JAX install guide to select the correct wheel for CUDA/ROCm where applicable.

## Validation Checklist

- Interpreter: `python -V` shows `3.12.x`.
- Import smoke test:
  ```
  python - <<'PY'
  import sys, acme
  print(sys.version)
  print('acme version OK')
  PY
  ```
- Dependency sanity: `pip check` (or `uv pip check`) reports no conflicts.
- Unit tests: `pytest -q acme` completes without failures on your platform.

## Troubleshooting

- Missing/broken wheels for binary packages (e.g., `dm-tree`, env libs):
  - Update `pip` and `setuptools`; retry install.
  - Ensure a working compiler toolchain if a source build is attempted.
  - If blockers persist, prefer a platform where wheels are available or a container image that matches the wheel matrix.

- Extras fail to resolve on 3.12:
  - Install the base package without the extra, then add components incrementally to isolate the offender.
  - For notebooks, use newer `ipykernel`.
  - For type checking, prefer `pyright` or a recent `pytype` if available on your platform.

## Notes for Maintainers

- `pyproject.toml` is the single source of truth for metadata and Python version support. Keep classifiers and `requires-python` in sync.
- Consider gating fragile extras with environment markers (e.g., `; python_version < "3.12"`) or upgrading pins where safe.
- The legacy `setup.py` can be retained for historical installers, but it should not diverge from `pyproject.toml` regarding supported Python versions.

