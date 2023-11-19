import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from pax.envs.iterated_matrix_game import (
    IteratedMatrixGame,
    EnvParams,
)

import jax
import jax.numpy as jnp

devices = jax.local_devices()
print(devices)
# batch over env initalisations
num_envs = 2
payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
rollout_length = 50

rng = jnp.concatenate(
    [jax.random.PRNGKey(0), jax.random.PRNGKey(1)]
).reshape(num_envs, -1)

env = IteratedMatrixGame(num_inner_steps=rollout_length, num_outer_steps=1)
env_params = EnvParams(payoff_matrix=payoff)

action = jnp.ones((num_envs,), dtype=jnp.float32)

# we want to batch over rngs, actions
env.step = jax.vmap(
    env.step,
    in_axes=(0, None, 0, None),
    out_axes=(0, None, 0, 0, 0),
)
env.reset = jax.vmap(
    env.reset, in_axes=(0, None), out_axes=(0, None))
obs, env_state = env.reset(rng, env_params)

# lets scan the rollout for speed
def rollout(carry, unused):
    last_obs, env_state, env_rng = carry
    actions = (action, action)
    obs, env_state, rewards, done, info = env.step(
        env_rng, env_state, actions, env_params
    )

    return (obs, env_state, env_rng), (
        obs,
        actions,
        rewards,
        done,
    )


final_state, trajectory = jax.lax.scan(
    rollout, (obs, env_state, rng), None, rollout_length
)

######################################################

from typing import NamedTuple

class EnvArgs(NamedTuple):
    env_id='iterated_matrix_game'
    runner='rl'
    num_envs=num_envs

class PPOArgs(NamedTuple):
    num_minibatches=10
    num_epochs=4
    gamma=0.96
    gae_lambda=0.95
    ppo_clipping_epsilon=0.2
    value_coeff=0.5
    clip_value=True
    max_gradient_norm=0.5
    anneal_entropy=True
    entropy_coeff_start=0.2
    entropy_coeff_horizon=5e5
    entropy_coeff_end=0.01
    lr_scheduling=True
    learning_rate=2.5e-2
    adam_epsilon=1e-5
    with_memory=False
    with_cnn=False
    hidden_size=16

######################################################

import jax.numpy as jnp
from pax.agents.ppo.ppo import make_agent

args = EnvArgs()
agent_args = PPOArgs()
agent = make_agent(args, 
    agent_args=agent_args,
    obs_spec=env.observation_space(env_params).n,
    action_spec=env.num_actions,
    seed=42,
    num_iterations=1e3,
    player_id=0,
    tabular=False,)


# batch MemoryState not TrainingState
agent.batch_reset = jax.jit(
    jax.vmap(agent.reset_memory, (0, None), 0),
    static_argnums=1,
)

agent.batch_policy = jax.jit(
    jax.vmap(agent._policy, (None, 0, 0), (0, None, 0))
)

agent.batch_update = jax.vmap(
    agent.update, (0, 0, None, 0), (None, 0, 0)
)


######################################################

from typing import NamedTuple

class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray

######################################################

def _rollout(carry, unused):
    """Runner for inner episode"""
    (
        rng,
        obs,
        a_state,
        a_mem,
        env_state,
        env_params,
    ) = carry
    # unpack rngs
    rngs = jax.random.split(rng, num_envs+1)
    rngs, rng = rngs[:-1], rngs[-1]

    action, a_state, new_a_mem = agent.batch_policy(
        a_state,
        obs[0],
        a_mem,
    )

    next_obs, env_state, rewards, done, info = env.step(
        rngs,
        env_state,
        (action, action),
        env_params,
    )

    traj = Sample(
        obs[0],
        action,
        rewards[0],
        new_a_mem.extras["log_probs"],
        new_a_mem.extras["values"],
        done,
        a_mem.hidden,
    )

    return (
        rng,
        next_obs,
        a_state,
        new_a_mem,
        env_state,
        env_params,
    ), traj
    
######################################################

rng = jax.random.PRNGKey(42)
init_hidden = jnp.zeros((agent_args.hidden_size))
rng, _rng = jax.random.split(rng)
a_state, a_memory = agent.make_initial_state(_rng, init_hidden)
rngs = jax.random.split(rng, num_envs)
obs, env_state = env.reset(rngs, env_params)

for _ in range(10):
    carry =  (rng, obs, a_state, a_memory, env_state, env_params)
    final_timestep, batch_trajectory = jax.lax.scan(
        _rollout,
        carry,
        None,
        10,
    )

    rng, obs, a_state, a_memory, env_state, env_params = final_timestep

    a_state, a_memory, stats = agent.update(
        batch_trajectory, obs[0], a_state, a_memory
    )