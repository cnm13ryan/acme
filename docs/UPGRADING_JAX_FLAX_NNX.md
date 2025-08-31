# Upgrading JAX and Migrating to Flax NNX

This repository is migrating JAX agents from Haiku/Sonnet to Flax NNX and
upgrading to a modern JAX stack. This note describes environment setup and
dependency guidance for heterogeneous GPUs (CUDA and ROCm).

## Target stack (JAX + Flax + friends) - Latest Versions

- JAX / jaxlib: **0.7.1** (released Aug 20, 2025 - Python 3.12 support)
- Flax: **0.11.2** (released Aug 28, 2025 - includes the `flax.nnx` API)
- Optax: **0.2.5** (released Jun 10, 2025)
- Chex: **0.1.90** (released Jul 23, 2025)
- RLax: >= 0.1.7

Notes:
- jaxlib is platform-specific. Install the CUDA or ROCm wheel that matches your
  drivers. See the official JAX installation docs for current wheel URLs.
- During migration, Haiku remains available for legacy agents; it will be
  removed after all agents are ported to Flax NNX.

## Installation (per node)

Create a Python 3.12 environment and then:

```bash
pip install -e .[envs]              # optional: environments
pip install -e .[jax_core,flax_nnx] # JAX + Flax NNX stack

# IMPORTANT: install the correct jaxlib for your GPU backend
# CUDA example (placeholder; use the wheel matching your CUDA toolchain):
pip install -U jax jaxlib            # with CUDA wheel for your CUDA version

# ROCm example (placeholder; use the ROCm wheel matching your ROCm release):
pip install -U jax jaxlib            # with ROCm wheel for your ROCm version

# Sanity check
python -c "import jax; print(jax.devices())"
```

If you have multiple GPU backends across nodes, keep the same `jax` version on
all nodes and install the matching `jaxlib` per node (CUDA vs ROCm).

## Lockfiles

After changing pins in `pyproject.toml`/`setup.py`, regenerate your environment
lock:

```bash
# Example commands; use your preferred solver
uv pip compile pyproject.toml -o uv.lock   # or
pip freeze > requirements.lock
```

## Migration approach

1. Introduce a thin Flax NNX compatibility shim to preserve the existing
   `init/apply` patterns used by builders and actors.
2. Pilot migration on one agent (e.g., DQN), validate parity on small tasks.
3. Iterate across remaining agents; remove Haiku when complete.

