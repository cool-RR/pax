# Pax
Here we start building Pax - a Jax Batched Environment for sequential matrix games (such as social dilemmas).

> *Pax (noun) - a period of peace that has been forced on a large area, such as an empire or even the whole world*

PAX has a simple interface, similar to [dm-env](https://github.com/deepmind/dm_env) and aspires to work with agents from both [acme](https://github.com/deepmind/acme) and [magi](https://github.com/ethanluoyc/magi).

```python
import IteratedPrisonersDilemma

env = IteratedPrisonersDilemma(episode_length, num_envs)
timesteps = env.reset()

agents = IndependentLeaners(
     agent_0,
     agent_1
)

while not timestep[0].last():
     actions = agents.step(timesteps)
     timestep = env.step(actions))
```

and timestep returns the following:

```python
timestep = timesteps[0]
timestep.observation.shape()
# (num_envs, num_states)
timestep.reward.shape()
# (num_envs, 1)
```

# Installation
Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to install JAX with the relevant accelerator support.

# Experiments
We present an example experimentation script at `pax/experiment.py`. We currently use `wandb` for logging.

```bash
python -m pax.experiment --wandb_group="testing"
```
