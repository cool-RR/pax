# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple JAX-based DQN implementation.
Reference: "Playing atari with deep reinforcement learning" (Mnih et al, 2015).
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.
"""

from typing import Any, Callable, NamedTuple, Sequence

from bsuite.baselines import base
from bsuite.environments import catch
from pax.dqn.replay_buffer import Replay

import logging
import wandb
import hydra
import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import distrax

class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: hk.Params
  target_params: hk.Params
  opt_state: Any
  step: int

class DQN(base.Agent):
  """A simple DQN agent using JAX."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: Callable[[jnp.ndarray], jnp.ndarray],
      optimizer: optax.GradientTransformation,
      batch_size: int,
      epsilon: float,
      rng: hk.PRNGSequence,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
  ):
    # Transform the (impure) network into a pure function.
    network = hk.without_apply_rng(hk.transform(network))

    # Define loss function.
    def loss(params: hk.Params,
             target_params: hk.Params,
             transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
      """Computes the standard TD(0) Q-learning loss on batch of transitions."""
      # observation, action, reward, discount, observation
      o_tm1, a_tm1, r_t, d_t, o_t = transitions
      q_tm1 = network.apply(params, o_tm1)
      q_t = network.apply(target_params, o_t)
      # TODO: Find a cleaner solution than squeezing. 
      a_tm1 = a_tm1.squeeze()
      r_t = r_t.squeeze()
      batch_q_learning = jax.vmap(rlax.q_learning)
      td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * d_t, q_t)
      return jnp.mean(td_error**2)

    # Define update function.
    @jax.jit
    def sgd_step(state: TrainingState,
                 transitions: Sequence[jnp.ndarray]) -> TrainingState:
      """Performs an SGD step on a batch of transitions."""
      gradients = jax.grad(loss)(state.params, state.target_params, transitions)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return TrainingState(
          params=new_params,
          target_params=state.target_params,
          opt_state=new_opt_state,
          step=state.step + 1)

    # Initialize the networks and optimizer.
    dummy_observation = np.zeros((1, obs_spec.num_values), jnp.float32)
    initial_params = network.init(next(rng), dummy_observation)
    initial_target_params = network.init(next(rng), dummy_observation)
    initial_opt_state = optimizer.init(initial_params)
    

    # This carries the agent state relevant to training.
    self._state = TrainingState(
        params=initial_params,
        target_params=initial_target_params,
        opt_state=initial_opt_state,
        step=0)
    self._sgd_step = sgd_step
    self._forward = jax.jit(network.apply)
    self._replay = Replay(capacity=replay_capacity)

    # Store hyperparameters.
    self._num_actions = action_spec.num_values
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon = epsilon
    self._total_steps = 0
    self._min_replay_size = min_replay_size

    self.train_steps = 0 
    self.eval_steps = 0 
    self.episodes = 0 
    self.num_target_updates = 0 

  def select_action(self, key, timestep: dm_env.TimeStep) -> jnp.array:
    """Selects batched actions according to an epsilon-greedy policy."""
    key, subkey = jax.random.split(key)
    if self.eval: # Evaluation mode is True 
        # Greedy policy, breaking ties uniformly at random.
        observation = timestep.observation[None, ...]
        q_values = self._forward(self._state.target_params, observation)
        greedy_dist = distrax.Greedy(q_values) #argument is preferences
        action = greedy_dist.sample(seed=key).reshape(-1, 1)
        self.eval_steps += 1
        return action
    else:
        # Epsilon greedy policy, breaking ties uniformly at random
        q_values = self._forward(self._state.params, timestep.observation) # (num_envs, state space)
        e_greedy_dist = distrax.EpsilonGreedy(preferences=q_values, epsilon=self._epsilon)
        actions = e_greedy_dist.sample(seed=key).reshape(-1, 1)
        assert actions.shape == (len(timestep.observation), 1)
        return actions

  def update(
      self,
      timestep: dm_env.TimeStep, 
      action: jnp.array,
      new_timestep: dm_env.TimeStep,
      key
  ):

    self._replay.add_batch([
        timestep.observation,
        action,
        new_timestep.reward,
        new_timestep.discount,
        new_timestep.observation],
        batch_size = len(timestep.observation) # num envs
    )

    # Increment number of training steps taken for plotting purposes
    self.train_steps +=1 
    self._total_steps += 1
    if self._total_steps % self._sgd_period != 0:
      return

    if self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD.
    transitions = self._replay.sample(self._batch_size, key)
    self._state = self._sgd_step(self._state, transitions)

    # Periodically update target parameters. 
    # Default update period is every 4 sgd steps. 
    if self._state.step % self._target_update_period == 0:
      self._state = self._state._replace(target_params=self._state.params)
      self.num_target_updates += 1

def default_agent(args, 
                  obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray
                  ) -> base.Agent:
  """Initialize a DQN agent with default parameters."""

  def network(inputs: jnp.ndarray) -> jnp.ndarray:
    flat_inputs = hk.Flatten()(inputs)
    # Single layer network which acts as a lookup table
    mlp = hk.Linear(action_spec.num_values, with_bias=False) 

    # Optimistic initializations (with constant weights)
    # mlp = hk.Linear(action_spec.num_values, w_init = hk.initializers.Constant(0.5), with_bias=False)

    # Larger MLP with a hidden layer of output size 10
    # mlp = hk.nets.MLP([10, action_spec.num_values], with_bias=False) #works alright with more layers
    action_values = mlp(flat_inputs)
    return action_values

  return DQN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=optax.adam(args.learning_rate),
      batch_size=args.batch_size,
      discount=args.discount,
      replay_capacity=args.replay_capacity,
      min_replay_size=args.min_replay_size,
      sgd_period=args.sgd_period,
      target_update_period=args.target_update_period,
      epsilon=args.epsilon,
      rng=hk.PRNGSequence(args.seed),
  )