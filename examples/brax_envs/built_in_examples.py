"""
Using example from:
https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb
"""

import time
import functools

import jax
from brax import envs

from braxviewer.WebViewerParallel import WebViewerParallel
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.envs.wrappers.viewer import ViewerWrapper
from etils import epath

env_name = 'humanoid_simplfied'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'mjx'  # @param ['generalized', 'positional', 'spring', 'mjx']

env_name_for_brax="humanoid" if env_name=='humanoid_simplfied' else env_name

env = envs.get_environment(env_name=env_name_for_brax,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

# Get XML string from brax environment assets
xml_string = epath.resource_path('brax') / f'envs/assets/{env_name_for_brax}.xml'
xml_string = xml_string.read_text()

def progress(num_steps, metrics):
  print(f'num_steps: {num_steps}')


if __name__ == "__main__":
    # Training functions with pre-configured hyperparameters
    train_fn = {
        'inverted_pendulum': functools.partial(ppo.train, num_timesteps=2_000_000, num_evals=20, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
        'inverted_double_pendulum': functools.partial(ppo.train, num_timesteps=20_000_000, num_evals=20, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
        'ant': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=10, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1),
        'humanoid_simplfied': functools.partial(ppo.train, num_timesteps=40000, num_evals=10, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=2, num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=4, batch_size=2, seed=1),
        'humanoid': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=10, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=10, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048, batch_size=1024, seed=1),
        'reacher': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=4, unroll_length=50, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048, batch_size=256, max_devices_per_host=8, seed=1),
        'humanoidstandup': functools.partial(ppo.train, num_timesteps=100_000_000, num_evals=20, reward_scaling=0.1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=15, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=6e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
        'hopper': functools.partial(sac.train, num_timesteps=6_553_600, num_evals=20, reward_scaling=30, episode_length=1000, normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=512, grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1),
        'walker2d': functools.partial(sac.train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128, grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1),
        'halfcheetah': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=1, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=0.001, num_envs=2048, batch_size=512, seed=3),
        'pusher': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=30, num_minibatches=16, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, batch_size=512, seed=3),
    }[env_name]
    
    # Get num_envs for viewer from the partial function
    num_envs = train_fn.keywords.get('num_envs', 8)  # default fallback
    
    # Create viewer for parallel environments
    viewer = WebViewerParallel(
        num_envs=num_envs,
        xml=xml_string,
    )
    viewer.run()

    # Create separate environments for training and evaluation
    env_for_evaluation = env  # Original environment for evaluation
    env_for_training = ViewerWrapper(env=env, viewer=viewer)  # Wrapped environment for training

    # Start training
    make_inference_fn, params, _ = train_fn(
        environment=env_for_training,
        eval_env=env_for_evaluation,
        progress_fn=progress,
    )

    print("--- Training Complete ---")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.stop()
        print("Program stopped.")
