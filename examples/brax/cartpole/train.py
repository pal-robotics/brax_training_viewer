"""A demonstration of batched environment training with a separated web viewer."""

import sys
import os
import time

# Add project path to sys.path to allow for relative imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import jax
import jax.numpy as jnp
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
from braxviewer.brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from braxviewer.wrapper import ViewerWrapper
from braxviewer.BraxSender import BraxSender
import config

# ==============================================================================
#  Environment Definition
# ==============================================================================

class CartPole(PipelineEnv):
    def __init__(self, backend: str = 'mjx', **kwargs):
        sys = mjcf.loads(config.XML_MODEL)
        self.step_dt = 0.02
        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend=backend, n_frames=n_frames)
    def reset(self, rng: jax.Array) -> State:
        key, theta_key, qd_key = jax.random.split(rng, 3)
        theta_init = jax.random.uniform(theta_key, (1,), minval=-0.1, maxval=0.1)[0]
        q_init = jnp.array([0.0, theta_init])
        qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)
        pipeline_state = self.pipeline_init(q_init, qd_init)
        reward, done = jnp.zeros(2)
        observation = self.get_observation(pipeline_state)
        metrics = {'rewards': reward}
        return State(pipeline_state=pipeline_state, obs=observation, reward=reward, done=done, metrics=metrics)
    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        observation = self.get_observation(pipeline_state)
        x, th = pipeline_state.q
        outside_x = jnp.abs(x) > 1.0
        outside_th = jnp.abs(th) > jnp.pi / 2
        done = jnp.float32(outside_x | outside_th)
        reward = jnp.cos(th)
        state.metrics.update({'rewards': reward})
        return state.replace(pipeline_state=pipeline_state, obs=observation, reward=reward, done=done)
    def get_observation(self, pipeline_state: State) -> jnp.ndarray:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])

# ==============================================================================
#  Training
# ==============================================================================

if __name__ == '__main__':
    env_for_evaluation = CartPole(backend='mjx')

    # The sender connects to the already-running WebViewer server.
    sender = BraxSender(
        host=config.HOST,
        port=config.PORT,
        xml=config.XML_MODEL,
        num_envs=config.NUM_PARALLEL_ENVS
    )
    sender.start()    
    # Wrap the environment with the ViewerWrapper to enable rendering.
    env_for_training = ViewerWrapper(env=env_for_evaluation, sender=sender)

    make_networks_factory = ppo_networks.make_ppo_networks

    def progress_fn(current_step, metrics):
        if current_step > 0:
            print(f'Training Step: {current_step} \t Eval Reward: {metrics["eval/episode_reward"]:.3f}')

    make_policy_fn, params, _ = ppo.train(
        environment=env_for_training,
        eval_env=env_for_evaluation,
        num_timesteps=config.NUM_TIMESTEPS,
        num_evals=config.NUM_EVALS,
        episode_length=config.EPISODE_LENGTH,
        num_envs=config.NUM_PARALLEL_ENVS,
        num_eval_envs=config.NUM_PARALLEL_ENVS,
        batch_size=config.BATCH_SIZE,
        num_minibatches=config.NUM_MINIBATCHES,
        unroll_length=config.UNROLL_LENGTH,
        num_updates_per_batch=config.NUM_UPDATES_PER_BATCH,
        normalize_observations=config.NORMALIZE_OBSERVATIONS,
        discounting=config.DISCOUNTING,
        learning_rate=config.LEARNING_RATE,
        entropy_cost=config.ENTROPY_COST,
        network_factory=make_networks_factory,
        seed=config.SEED,
        wrap_env=True,
        progress_fn=progress_fn,
    )

    print("--- Training Complete ---")
    print("Stopping sender...")
    sender.stop()
    print("Sender stopped.")