# Running Acme Agents: Complete Guide

This guide provides detailed instructions on how to instantiate and run different Acme agents across various environments, based on the working examples in the `examples/` directory.

## Overview of Available Agents

Acme provides a rich ecosystem of reinforcement learning agents across different categories:

### Continuous Control Agents
- **D4PG** - Deterministic Policy Gradient with distributional critic
- **PPO** - Proximal Policy Optimization  
- **SAC** - Soft Actor-Critic
- **TD3** - Twin Delayed Deep Deterministic Policy Gradient
- **DMPO** - Distributional Maximum a Posteriori Policy Optimization
- **MPO** - Maximum a Posteriori Policy Optimization
- **WPO** - Wasserstein Policy Optimization

### Discrete Control Agents
- **DQN** - Deep Q-Network
- **Rainbow DQN** - DQN with multiple improvements
- **Quantile Regression DQN** - Distributional DQN
- **IMPALA** - Scalable distributed deep-RL
- **R2D2** - Recurrent Experience Replay
- **MuZero** - Model-based planning agent

### Offline/Imitation Learning
- **BC** - Behavior Cloning
- **BCQ** - Batch-Constrained Q-learning  
- **CQL** - Conservative Q-Learning
- **CRR** - Critic Regularized Regression

## Pattern 1: High-Level Experiment Framework

The most common and recommended approach uses Acme's experiment framework:

### Basic Structure
```python
from acme.jax import experiments
from acme.agents.jax import [AGENT_NAME]

def build_experiment_config():
    # 1. Configure the agent
    config = [AGENT_NAME].[AGENT_NAME]Config(
        learning_rate=3e-4,
        # ... other hyperparameters
    )
    builder = [AGENT_NAME].[AGENT_NAME]Builder(config)
    
    # 2. Define environment factory
    def env_factory(seed):
        return make_environment(env_name)
    
    # 3. Define network factory  
    def network_factory(spec):
        return [AGENT_NAME].make_networks(spec, hidden_sizes)
    
    return experiments.ExperimentConfig(
        builder=builder,
        environment_factory=env_factory, 
        network_factory=network_factory,
        seed=0,
        max_num_actor_steps=1_000_000
    )

# Run the experiment
experiments.run_experiment(
    experiment=build_experiment_config(),
    eval_every=50_000,
    num_eval_episodes=10
)
```

## Pattern 2: Running Different Agents

### D4PG on Continuous Control

```python
from acme.agents.jax import d4pg
from acme.jax import experiments
import gymnasium as gym
from acme import wrappers

def make_gym_environment(env_name):
    """Create a Gym environment wrapped for Acme."""
    env = gym.make(env_name)
    env = wrappers.GymWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

def run_d4pg_on_halfcheetah():
    """Run D4PG on HalfCheetah environment."""
    
    # Configure D4PG
    d4pg_config = d4pg.D4PGConfig(
        learning_rate=3e-4,
        sigma=0.2,           # Exploration noise
        discount=0.99,
        target_update_period=100,
        min_replay_size=1000,
        max_replay_size=1_000_000,
        batch_size=256
    )
    
    def network_factory(spec):
        return d4pg.make_networks(
            spec,
            policy_layer_sizes=(256, 256, 256),
            critic_layer_sizes=(256, 256, 256),
            vmin=-1000.0,  # Value bounds for distributional critic
            vmax=1000.0
        )
    
    config = experiments.ExperimentConfig(
        builder=d4pg.D4PGBuilder(d4pg_config),
        environment_factory=lambda seed: make_gym_environment('HalfCheetah-v4'),
        network_factory=network_factory,
        seed=0,
        max_num_actor_steps=1_000_000
    )
    
    experiments.run_experiment(
        experiment=config,
        eval_every=50_000,
        num_eval_episodes=10
    )
```

### DQN on Atari

```python
from acme.agents.jax import dqn
from acme.agents.jax.dqn import losses
import gymnasium as gym
from acme import wrappers
import functools

def make_atari_environment(level='Pong'):
    """Create an Atari environment with proper preprocessing."""
    env = gym.make(f'{level}NoFrameskip-v4', full_action_space=True)
    
    wrapper_list = [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            scale_dims=(84, 84),
            to_float=True,
            max_episode_len=108_000,
            num_stacked_frames=4,
            grayscaling=True,
            zero_discount_on_life_loss=False,
        ),
        wrappers.SinglePrecisionWrapper,
    ]
    
    return wrappers.wrap_all(env, wrapper_list)

def run_dqn_on_atari():
    """Run DQN on Atari Pong."""
    
    # Configure DQN
    config = dqn.DQNConfig(
        discount=0.99,
        eval_epsilon=0.05,     # Epsilon for evaluation
        learning_rate=5e-5,
        n_step=1,
        epsilon=0.01,          # Training epsilon  
        target_update_period=2000,
        min_replay_size=20_000,
        max_replay_size=1_000_000,
        batch_size=32
    )
    
    loss_fn = losses.QLearning(
        discount=config.discount, 
        max_abs_reward=1.0
    )
    
    def network_factory(spec):
        # Creates CNN networks for Atari
        import haiku as hk
        from acme.jax import networks as networks_lib
        from acme.jax import utils
        
        def network(inputs):
            model = hk.Sequential([
                networks_lib.AtariTorso(),  # CNN for Atari frames
                hk.nets.MLP([512, spec.actions.num_values])
            ])
            return model(inputs)
        
        network_hk = hk.without_apply_rng(hk.transform(network))
        obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
        network = networks_lib.FeedForwardNetwork(
            init=lambda rng: network_hk.init(rng, obs), 
            apply=network_hk.apply
        )
        typed_network = networks_lib.non_stochastic_network_to_typed(network)
        return dqn.DQNNetworks(policy_network=typed_network)
    
    config = experiments.ExperimentConfig(
        builder=dqn.DQNBuilder(config, loss_fn=loss_fn),
        environment_factory=lambda seed: make_atari_environment('Pong'),
        network_factory=network_factory,
        seed=0,
        max_num_actor_steps=1_000_000
    )
    
    experiments.run_experiment(experiment=config)
```

### PPO with Custom Networks

```python
from acme.agents.jax import ppo

def run_ppo_on_cartpole():
    """Run PPO on CartPole with custom configuration."""
    
    config = ppo.PPOConfig(
        learning_rate=3e-4,
        entropy_cost=1e-3,
        value_cost=0.5,
        max_gradient_norm=0.5,
        discount=0.99,
        gae_lambda=0.95,
        unroll_length=32,
        num_minibatches=4,
        num_epochs=4,
        normalize_advantage=True,
        normalize_value=True,
        obs_normalization_fns_factory=ppo.build_mean_std_normalizer
    )
    
    def network_factory(spec):
        return ppo.make_networks(
            spec, 
            policy_layer_sizes=(64, 64),
            value_layer_sizes=(64, 64)
        )
    
    experiment_config = experiments.ExperimentConfig(
        builder=ppo.PPOBuilder(config),
        environment_factory=lambda seed: make_gym_environment('CartPole-v1'),
        network_factory=network_factory,
        seed=0,
        max_num_actor_steps=500_000
    )
    
    experiments.run_experiment(
        experiment=experiment_config,
        eval_every=25_000,
        num_eval_episodes=10
    )
```

## Pattern 3: Different Environment Types

### Gymnasium Environments
```python
def make_gym_environment(env_name):
    import gymnasium as gym
    env = gym.make(env_name)
    env = wrappers.GymWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

# Examples:
# - make_gym_environment('CartPole-v1')       # Discrete
# - make_gym_environment('HalfCheetah-v4')    # Continuous  
# - make_gym_environment('Pong-v4')           # Atari
```

### DeepMind Control Suite
```python
def make_control_environment(domain, task):
    from dm_control import suite as dm_suite
    env = dm_suite.load(domain, task)
    env = wrappers.ConcatObservationWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

# Examples:
# - make_control_environment('cartpole', 'balance')
# - make_control_environment('cheetah', 'run') 
# - make_control_environment('walker', 'walk')
```

### Custom Environments
```python
import dm_env
from acme import specs
import numpy as np

class MyCustomEnvironment(dm_env.Environment):
    def __init__(self):
        self._step_count = 0
        self._episode_ended = False
    
    def reset(self):
        self._step_count = 0
        self._episode_ended = False
        return dm_env.restart(self._get_observation())
    
    def step(self, action):
        if self._episode_ended:
            return self.reset()
        
        self._step_count += 1
        reward = self._get_reward(action)
        observation = self._get_observation()
        
        if self._step_count >= 200:  # Max episode length
            self._episode_ended = True
            return dm_env.termination(reward, observation)
        
        return dm_env.transition(reward, observation)
    
    def observation_spec(self):
        return specs.Array(shape=(4,), dtype=np.float32)
    
    def action_spec(self):
        return specs.DiscreteArray(num_values=2)
    
    def _get_observation(self):
        return np.random.rand(4).astype(np.float32)
    
    def _get_reward(self, action):
        return 1.0 if action == 0 else -1.0
```

## Pattern 4: Distributed Training

```python
from acme.utils import lp_utils
import launchpad as lp

def run_distributed_d4pg():
    """Run D4PG in distributed mode with multiple actors."""
    
    config = build_experiment_config()  # Same as before
    
    # Create distributed program
    program = experiments.make_distributed_experiment(
        experiment=config,
        num_actors=8,                    # Number of parallel actors
        environment_factory=env_factory, # Same environment factory
        network_factory=network_factory, # Same network factory
        evaluator_factories=[]           # Optional custom evaluators
    )
    
    # Launch with appropriate resources
    lp.launch(
        program, 
        xm_resources=lp_utils.make_xm_docker_resources(program)
    )
```

## Pattern 5: Offline Learning

```python
from acme.agents.jax import bc
import tensorflow as tf
import jax
import optax

def run_behavior_cloning():
    """Run Behavior Cloning on demonstration data."""
    
    # Load your dataset
    def make_demonstrations():
        # Your dataset loading logic here
        # Should return tf.data.Dataset with (observation, action, reward, ...)
        dataset = load_expert_demonstrations()
        return dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    
    environment = make_environment()
    environment_spec = specs.make_environment_spec(environment)
    
    # Create networks
    bc_networks = bc.make_networks(
        environment_spec,
        policy_layer_sizes=(256, 256),
    )
    
    # Configure learner
    learner = bc.BCLearner(
        networks=bc_networks,
        random_key=jax.random.PRNGKey(0),
        loss_fn=bc.logp(),  # Log probability loss
        optimizer=optax.adam(1e-3),
        dataset=make_demonstrations()
    )
    
    # Training loop
    for step in range(100_000):
        learner.step()
        
        if step % 1000 == 0:
            # Evaluate the policy
            evaluate_policy(learner, environment)
```

## Pattern 6: Custom Configurations & Hyperparameters

### Hyperparameter Tuning Example
```python
def run_hyperparameter_sweep():
    """Run experiments with different hyperparameters."""
    
    learning_rates = [1e-4, 3e-4, 1e-3]
    layer_sizes = [(64, 64), (256, 256), (512, 512)]
    
    for lr in learning_rates:
        for layers in layer_sizes:
            print(f"Running with lr={lr}, layers={layers}")
            
            config = d4pg.D4PGConfig(
                learning_rate=lr,
                sigma=0.2,
                discount=0.99
            )
            
            def network_factory(spec):
                return d4pg.make_networks(
                    spec,
                    policy_layer_sizes=layers,
                    critic_layer_sizes=layers
                )
            
            experiment_config = experiments.ExperimentConfig(
                builder=d4pg.D4PGBuilder(config),
                environment_factory=lambda seed: make_gym_environment('HalfCheetah-v4'),
                network_factory=network_factory,
                seed=42,
                max_num_actor_steps=100_000  # Shorter for sweep
            )
            
            experiments.run_experiment(experiment_config)
```

## Pattern 7: Custom Logging & Evaluation

```python
from acme.utils import loggers
import collections

def run_with_custom_logging():
    """Run experiment with detailed logging."""
    
    # Create logger that stores data in memory
    logger_dict = collections.defaultdict(loggers.InMemoryLogger)
    
    def logger_factory(name, steps_key=None, task_id=None):
        return logger_dict[name]
    
    experiment_config = experiments.ExperimentConfig(
        builder=d4pg.D4PGBuilder(d4pg.D4PGConfig()),
        environment_factory=lambda seed: make_gym_environment('CartPole-v1'),
        network_factory=lambda spec: d4pg.make_networks(spec),
        logger_factory=logger_factory,  # Add custom logger
        seed=0,
        max_num_actor_steps=100_000
    )
    
    experiments.run_experiment(experiment_config)
    
    # Access logged data
    training_data = logger_dict['learner'].data
    eval_data = logger_dict['evaluator'].data
    
    # Plot results
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(eval_data)
    plt.plot(df['actor_steps'], df['episode_return'])
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Return')
    plt.title('Training Progress')
    plt.show()
```

## Running the Examples

### Command Line Execution
```bash
# Run D4PG on HalfCheetah
python examples/baselines/rl_continuous/run_d4pg.py \
    --env_name=gym:HalfCheetah-v4 \
    --num_steps=1000000 \
    --run_distributed=false

# Run DQN on Atari  
python examples/baselines/rl_discrete/run_dqn.py \
    --env_name=Pong \
    --num_steps=1000000 \
    --run_distributed=false

# Run PPO on continuous control
python examples/baselines/rl_continuous/run_ppo.py \
    --env_name=gym:CartPole-v1 \
    --num_steps=500000
```

### Programmatic Execution
```python
if __name__ == '__main__':
    # Choose your experiment
    run_d4pg_on_halfcheetah()
    # run_dqn_on_atari() 
    # run_ppo_on_cartpole()
```

## Quick Start Summary

1. **Choose an Agent**: Pick from D4PG, PPO, DQN, SAC, etc.
2. **Configure Agent**: Set hyperparameters using `AgentConfig`
3. **Create Environment Factory**: Define how to make your environment
4. **Create Network Factory**: Define neural network architectures  
5. **Build Experiment Config**: Combine components
6. **Run Experiment**: Use `experiments.run_experiment()`

This framework provides maximum flexibility while handling the complexity of distributed training, logging, evaluation, and checkpointing automatically!

## Next Steps

- Explore the `examples/quickstart.ipynb` notebook for hands-on tutorial
- Check `examples/tutorial.ipynb` for deep dive into agent architecture  
- Browse `examples/baselines/` for production-ready implementations
- Look at `examples/offline/` for offline learning examples

The examples directory contains battle-tested implementations that you can use as templates for your own experiments!

## Common Agent Configurations

### Agent Configuration Quick Reference

| Agent | Best For | Key Hyperparameters |
|-------|----------|-------------------|
| **D4PG** | Continuous control | `learning_rate=3e-4`, `sigma=0.2` |
| **PPO** | Sample efficiency | `learning_rate=3e-4`, `entropy_cost=1e-3` |
| **DQN** | Discrete actions | `learning_rate=5e-5`, `epsilon=0.01` |
| **SAC** | Continuous, maximum entropy | `learning_rate=3e-4`, `temperature=0.2` |
| **BC** | Offline learning | `learning_rate=1e-3`, dataset required |

### Environment Quick Reference

| Environment Type | Factory Function | Example |
|-----------------|------------------|---------|
| **Gymnasium** | `make_gym_environment()` | CartPole-v1, HalfCheetah-v4 |
| **DeepMind Control** | `make_control_environment()` | cartpole:balance, cheetah:run |
| **Atari** | `make_atari_environment()` | Pong, Breakout, Space Invaders |
| **Custom** | Inherit from `dm_env.Environment` | Your custom task |

This guide covers the most common patterns found in the Acme examples directory and should serve as your primary reference for running experiments!