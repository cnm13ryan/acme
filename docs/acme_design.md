# Acme Design Document — Background and Design Decisions

Last updated: 2025-08-31

## Executive Summary

Acme is a research-focused reinforcement learning (RL) framework that provides simple, modular building blocks for composing agents that scale from single‑process experiments to large distributed systems. Its core architectural choices—clean separation of acting and learning, a first‑class experience storage system (Reverb), a programming model for distributed execution (Launchpad), and a standardized dataset ecosystem (RLDS)—aim to make RL algorithms readable, reproducible, and scalable without rewriting the core agent code across scales. This document consolidates the background and the rationale behind those choices and maps them to the code organization used in this repository.

## Background and Prior Art

- Acme originated as a library of RL components and reference agents designed to bridge small‑scale and distributed training with the same abstractions and code paths. It emphasizes readability, simplicity, and reproducibility for researchers while retaining performance characteristics necessary for large‑scale experiments.
- Reverb is a high‑throughput, client–server system for experience replay and generic ML data transport with configurable sampling/removal strategies and built‑in rate limiting to control sampling‑to‑insertion ratios.
- Launchpad is a programming model that expresses a distributed program as a directed graph of service nodes with explicit RPC edges and pluggable launch backends (local multiprocessing/threads, multi‑host, cloud), simplifying distributed orchestration for ML/RL.
- RLDS standardizes how sequential decision‑making data (episodes/steps) are recorded, shared, transformed, and consumed, enabling offline RL, imitation learning, and dataset interchange with minimal loss of information.

These systems were designed to interoperate: Acme agents use Reverb for experience storage and Launchpad for process orchestration; offline/interactive data can be exchanged using RLDS converters and schemas.

## Goals and Non‑Goals

### Goals
- Reproducibility: deterministic seeds, explicit component boundaries, and reference implementations of common agents.
- Readability and simplicity: minimal “framework magic”; code closely mirrors algorithm descriptions.
- Scalability: the same agent composition runs single‑process or distributed by swapping orchestration/topology, not agent logic.
- Extensibility: easy to author novel agents by composing standard building blocks (e.g., adders, learners, replay, networks).
- Interoperability: standardized data exchange through RLDS and decoupled storage through Reverb.

### Non‑Goals
- Serving/production platform: Acme focuses on research workflows rather than production inference serving.
- Full MLOps stack: experiment tracking, artifact registries, and cluster schedulers are intentionally out of scope beyond Launchpad integration.

## Repository Orientation

The repository is organized to keep reusable abstractions close to their tests and examples:

- `acme/agents/`: Reference agents and agent builders (e.g., DQN/R2D2, SAC/D4PG, MPO) for both single‑process and distributed settings.
- `acme/adders/`: “Adders” transform online interactions (env transitions, n‑step traces, trajectories) into items inserted into replay/datasets.
- `acme/environment_loops/`: Canonical interaction loops driving `agent` ↔ `environment` execution, evaluation, and logging.
- `acme/utils/`: Utilities for checkpointing, schedules, logging/metrics, tree utilities, JAX/TF helpers, etc.
- `acme/wrappers/`: Environment wrappers for observation/action normalization, frame stacking, time‑limit handling, etc.
- `acme/jax/`, `acme/tf/`: Backend‑specific network modules and learners.
- `examples/`: Runnable examples and baselines; good starting points for composing agents and loops.
- `docs/`: Supplemental docs (this file belongs here).

Note: Paths reflect the typical Acme layout; when integrating with other infrastructure, keep module boundaries aligned with the abstractions below.

## Architectural Overview

Acme’s core is a small set of composable abstractions that factor an RL agent into testable units and explicit process roles. The same abstractions work in single‑process and distributed topologies.

### Core Concepts

- Environment: Any API compatible with the project’s wrappers (e.g., dm_control, Gym/Gymnasium, custom envs). Produces observations/rewards/discounts.
- Actor: Policy application in an environment. Owns exploration behavior, episodic state, and “adder” hooks to write experience.
- Learner: Optimizes the agent’s parameters from batches sampled from storage (e.g., Reverb) or offline datasets (RLDS). Handles loss computation, updates, and checkpointing.
- Evaluator: Runs evaluation policies/episodes, typically without exploration noise, logs metrics.
- Networks: Backend‑specific function approximators (JAX/TF) and policy/value heads; separate from learners for testability.
- Adders: Adapt online interactions into replay items (e.g., transition, N‑step, sequence adders), including priorities/weights metadata.
- Replay/Storage: Reverb server(s) hosting one or more tables (sampler, remover, rate‑limiter, capacity, max times sampled). Provides client ops for insert/sample.
- Builders/Configs: Declarative composition of the above into a concrete agent (e.g., which adder, what replay table, learner hyperparameters).
- Environment Loop(s): Orchestrate `actor.step()`, adder writes, periodic learner updates, evaluations, and logging cadence.

### Process Topologies

1) Single‑process: Actor, learner, and replay client operate in one process; storage may be in‑process or a local Reverb server.
2) Multi‑process same‑host: Separate processes for actors, learner, and a local Reverb server to avoid GIL/contention.
3) Distributed multi‑host: Launchpad program describes a DAG of nodes (actors, learner, replay server, evaluators); edges define RPC channels, and the launch backend maps nodes to machines/containers.

The same `AgentBuilder` and agent code are reused; only the program topology and launch configuration change.

## Data and Storage Design

### Reverb Tables

Reverb provides one or more tables per server. Each table defines:

- `sampler`: selection strategy (e.g., Uniform, Prioritized, Heap‑based, Selector‑based sequences).
- `remover`: eviction policy (e.g., FIFO, LIFO, Heap).
- `rate_limiter`: sampling/insert constraints (e.g., KeepMinSize, SamplesPerInsert), back‑pressuring actors or learners when under/over‑provisioned.
- Capacity and `max_times_sampled`: bounding memory and staleness.

Items reference underlying data elements; multiple items may share elements, and elements are deleted when no item references remain. This model enables multiple views (e.g., PER for 2‑step transitions and FIFO for 3‑step sequences) over the same underlying experience without duplication.

Design implications:

- Stability: Rate limiters enforce sample‑to‑insert ratios that prevent degenerate training regimes (e.g., over‑sampling fresh or stale data).
- Flexibility: Different tables can feed different learners/components (policy/value heads, auxiliary models) concurrently.
- Performance: Client–server architecture scales to thousands of concurrent clients; colocating server with learners reduces latency, while remote servers decouple compute from memory pressure.

### Adders and Item Shapes

Adders define the granularity of what is inserted into storage:

- Transition adders (s, a, r, γ, s′) for value‑based/actor‑critic.
- N‑step adders for bootstrapped targets and multi‑step returns.
- Sequence/trajectory adders for RNNs or temporal credit assignment.

Priorities and auxiliary metadata are attached at insertion time (e.g., TD‑error‑derived priorities), enabling prioritized replay and importance weighting in the learner.

### RLDS Integration

RLDS standardizes episodic data with step‑level keys such as `observation`, `action`, `reward`, `discount`, and boolean indicators like `is_first`, `is_last`, `is_terminal`. Converters bridge Reverb streams to RLDS episodes and vice versa so that offline datasets can feed Acme learners without bespoke preprocessing and interactive agents can export reproducible datasets.

## Distributed Orchestration with Launchpad

Launchpad expresses the program as a graph of typed nodes with explicit communication edges. A single description can launch locally (threads/processes), on multi‑host clusters, or on cloud backends with resource specifications (CPU, RAM, accelerators). For RL, canonical nodes include many Actors, one or more Learners, a Reverb Server, and Evaluators. This explicit topology makes scaling, fault isolation, and resource placement a matter of configuration rather than invasive code changes.

Operational notes:

- Actors often colocate lightweight preprocessing and adders to minimize network hops for write‑heavy paths.
- Learners colocate with Reverb servers for low‑latency sampling and high batch throughput.
- Evaluators are decoupled to prevent exploration noise or parameter churn from polluting evaluation metrics.

## Algorithm Templates and Reference Agents

Acme ships reference implementations that double as design “executable specs.” Common patterns include:

- Value‑based: DQN, Rainbow, R2D2, with PER/FIFO tables and sequence adders for RNN agents.
- Actor‑critic: DDPG, TD3, SAC, D4PG with uniform replay and target networks.
- Policy search: MPO and variants with off‑policy replay and KL‑regularized updates.
- On‑policy via queues: FIFO tables used as on‑policy queues for PPO‑style updates when desired.

These share: (1) an `AgentBuilder` that wires adders, replay tables, learners, and networks; (2) actors that differ mainly in exploration and policy application; and (3) learners that differ in loss functions, target computations, and update rules.

## Configuration, Reproducibility, and Experimentation

- Determinism: Seed all RNGs (Python/JAX/TF/env), fix dataset shuffles, and checkpoint both learner state and replay cursors.
- Config surfaces: Batch sizes, unroll lengths, target update periods, replay table capacities/ratios, optimizer schedules.
- Sweepability: Keep configs declarative to enable grid/Random/BO sweeps without code edits.
- Checkpointing: Save learner parameters, optimizer state, and experiment metadata; keep snapshots small and atomic.

## Reliability, Observability, and Operations

- Health checks: Reverb server liveness and table health (size, samples/insert, dropped/waiting requests) exposed via metrics/logs.
- Backpressure: Rely on Reverb rate‑limiters to throttle actors/learners; surface blocked durations and queue depths.
- Logging/metrics: Standardize episode returns, lengths, learner losses, replay stats (age, staleness), wall‑clock throughput (env steps/sec, samples/sec), and update rates.
- Fault tolerance: Stateless actors restart cheaply; learners recover from checkpoints; Reverb can rebuild tables from upstream data if configured to persist.

## Performance Considerations

- Throughput: Colocate learners with replay for hot sampling paths; batch RPCs where supported.
- Latency hiding: Many actors amortize environment/inference latency; learners prefer larger batch updates with prefetch pipelines.
- Stability: Tune `SamplesPerInsert` and `MinSize` rate‑limiters; cap `max_times_sampled` to prevent stale over‑recycling; prioritize by TD‑error where helpful with importance weights to debias.
- Memory: Prefer sequence compression and shared data elements across items; prune auxiliary tables aggressively.

## Extensibility Patterns

- New agent: Implement a `Learner` (loss/update), choose/add a `Network` architecture, pick/create an `Adder`, define replay `Table`(s), and wire with an `AgentBuilder`.
- New storage behavior: Compose a `Table` with desired `sampler`, `remover`, `rate_limiter`, and capacity policy; adjust adder item shapes accordingly.
- Offline/IL: Use RLDS loaders to feed learners; export interactive data to RLDS for reproducibility and sharing.

## Security and Privacy

- Data handling: Classify datasets; avoid storing raw PII in replay buffers; encrypt persisted datasets where needed; prefer ephemeral in‑memory tables for sensitive data.
- Least privilege: Scope networking ports for Reverb/Launchpad to trusted networks; avoid accidental data exfiltration from debug endpoints.

## Testing Strategy

- Unit tests co‑located with modules (`*_test.py`).
- Deterministic smoke tests for environment loops and adders (shape/typing, n‑step returns, episode boundaries).
- IO tests: Replay table contracts (sampler/remover/rate‑limit invariants) and RLDS converters.
- Integration tests: Small‑scale training runs with fixed seeds to validate learning curves/regression protection.

## Rollout and Migration

1) Start single‑process experiments to validate logic and hyperparameters.
2) Move to multi‑process same‑host to separate actors/learner and stabilize IO.
3) Switch to a Launchpad multi‑host topology when scaling actors or learners; tune Reverb tables and rate‑limiters for new throughput regimes.
4) Persist/export datasets via RLDS for offline analysis and reproducibility.

## Key Design Decisions and Trade‑offs

- Acting vs. Learning split: Enables parallelism and reuse across agents; introduces staleness between actor policy and learner parameters—mitigated with periodic parameter pushes and importance corrections where needed.
- Client–server replay: Decouples compute from storage, supports many clients, and enables sophisticated sampling policies; adds network hops (mitigate via colocation and batching).
- Explicit program graphs (Launchpad): Improves clarity/scalability and ease of topology changes; adds a dependency on a specific orchestration model (kept pluggable across backends).
- Standardized datasets (RLDS): Increases reproducibility and sharing; requires careful converters to preserve all semantics (e.g., discounts, terminal markers, time‑limits).

## Glossary

- Adder: Component that converts online interaction into replay items with optional priorities/weights.
- Reverb Table: Storage unit with sampler/remover/limiter policies.
- SamplesPerInsert: Target ratio controlling training data freshness vs. throughput.
- PER: Prioritized Experience Replay; bias sampling by priority with importance weights in the learner.
- RLDS: Reinforcement Learning Datasets; standardized episodic schema for SDM.
- Launchpad Program: A DAG of nodes (processes) with typed RPC edges describing a distributed application.

## References (canonical sources)

1) Hoffman et al., “Acme: A Research Framework for Distributed Reinforcement Learning,” arXiv:2006.00979, 2020.
2) Cassirer et al., “Reverb: A Framework for Experience Replay,” arXiv:2102.04736, 2021.
3) Yang et al., “Launchpad: A Programming Model for Distributed Machine Learning Research,” arXiv:2106.04516, 2021.
4) Ramos et al., “RLDS: an Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning,” arXiv:2111.02767, 2021.

