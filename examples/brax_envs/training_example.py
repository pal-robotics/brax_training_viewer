"""
Using example from:
https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb
"""

#TODO

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
from brax import envs

from braxviewer.WebViewer import WebViewer
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

env_name = 'humanoid'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'positional'  # @param ['generalized', 'positional', 'spring']

env_name_for_brax="humanoid" if env_name=='humanoid_simplfied' else env_name

env = envs.get_environment(env_name=env_name_for_brax,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

episode_length = 1000 
num_timesteps = 40000
batch_size = 2
num_minibatches = 2
num_envs = batch_size * num_minibatches

def progress(num_steps, metrics):
  print(f'num_steps: {num_steps}')


if __name__ == "__main__":
    viewer = WebViewer()
    viewer.run()
    viewer.init(env)

    if env_name == 'inverted_pendulum':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=2_000_000, 
            num_evals=20, 
            reward_scaling=10, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=5, 
            num_minibatches=32, 
            num_updates_per_batch=4, 
            discounting=0.97, 
            learning_rate=3e-4, 
            entropy_cost=1e-2, 
            num_envs=2048, 
            batch_size=1024, 
            seed=1
        )
    elif env_name == 'inverted_double_pendulum':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=20_000_000, 
            num_evals=20, 
            reward_scaling=10, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=5, 
            num_minibatches=32, 
            num_updates_per_batch=4, 
            discounting=0.97, 
            learning_rate=3e-4, 
            entropy_cost=1e-2, 
            num_envs=2048, 
            batch_size=1024, 
            seed=1
        )
    elif env_name == 'ant':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=50_000_000, 
            num_evals=10, 
            reward_scaling=10, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=5, 
            num_minibatches=32, 
            num_updates_per_batch=4, 
            discounting=0.97, 
            learning_rate=3e-4, 
            entropy_cost=1e-2, 
            num_envs=4096, 
            batch_size=2048, 
            seed=1
        )
    elif env_name == 'humanoid_simplfied':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=num_timesteps, 
            num_evals=10, 
            reward_scaling=0.1, 
            episode_length=episode_length, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=10, 
            num_minibatches=num_minibatches, 
            num_updates_per_batch=8, 
            discounting=0.97, 
            learning_rate=3e-4, 
            entropy_cost=1e-3, 
            num_envs=num_envs, 
            batch_size=batch_size, 
            seed=1
        )
    elif env_name == 'humanoid':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=50_000_000, 
            num_evals=10, 
            reward_scaling=0.1, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=10, 
            num_minibatches=32, 
            num_updates_per_batch=8, 
            discounting=0.97, 
            learning_rate=3e-4, 
            entropy_cost=1e-3, 
            num_envs=2048, 
            batch_size=1024, 
            seed=1
        )
    elif env_name == 'reacher':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=50_000_000, 
            num_evals=20, 
            reward_scaling=5, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=4, 
            unroll_length=50, 
            num_minibatches=32, 
            num_updates_per_batch=8, 
            discounting=0.95, 
            learning_rate=3e-4, 
            entropy_cost=1e-3, 
            num_envs=2048, 
            batch_size=256, 
            max_devices_per_host=8, 
            seed=1
        )
    elif env_name == 'humanoidstandup':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=100_000_000, 
            num_evals=20, 
            reward_scaling=0.1, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=15, 
            num_minibatches=32, 
            num_updates_per_batch=8, 
            discounting=0.97, 
            learning_rate=6e-4, 
            entropy_cost=1e-2, 
            num_envs=2048, 
            batch_size=1024, 
            seed=1
        )
    elif env_name == 'hopper':
        make_inference_fn, params, _ = sac.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=6_553_600, 
            num_evals=20, 
            reward_scaling=30, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            discounting=0.997, 
            learning_rate=6e-4, 
            num_envs=128, 
            batch_size=512, 
            grad_updates_per_step=64, 
            max_devices_per_host=1, 
            max_replay_size=1048576, 
            min_replay_size=8192, 
            seed=1
        )
    elif env_name == 'walker2d':
        make_inference_fn, params, _ = sac.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=7_864_320, 
            num_evals=20, 
            reward_scaling=5, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            discounting=0.997, 
            learning_rate=6e-4, 
            num_envs=128, 
            batch_size=128, 
            grad_updates_per_step=32, 
            max_devices_per_host=1, 
            max_replay_size=1048576, 
            min_replay_size=8192, 
            seed=1
        )
    elif env_name == 'halfcheetah':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=50_000_000, 
            num_evals=20, 
            reward_scaling=1, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=20, 
            num_minibatches=32, 
            num_updates_per_batch=8, 
            discounting=0.95, 
            learning_rate=3e-4, 
            entropy_cost=0.001, 
            num_envs=2048, 
            batch_size=512, 
            seed=3
        )
    elif env_name == 'pusher':
        make_inference_fn, params, _ = ppo.train(
            environment=env, 
            progress_fn=progress, 
            viewer=viewer,
            num_timesteps=50_000_000, 
            num_evals=20, 
            reward_scaling=5, 
            episode_length=1000, 
            normalize_observations=True, 
            action_repeat=1, 
            unroll_length=30, 
            num_minibatches=16, 
            num_updates_per_batch=8, 
            discounting=0.95, 
            learning_rate=3e-4,
            entropy_cost=1e-2, 
            num_envs=2048, 
            batch_size=512, 
            seed=3
        )
