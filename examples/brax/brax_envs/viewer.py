"""A script to run the WebViewer server independently for the Brax environments example."""

import time
import sys
import os

# Add project path to allow for relative imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from braxviewer.WebViewer import WebViewer
from etils import epath
import config

def get_env_name_for_brax(env_name):
    """Convert environment name for Brax compatibility."""
    return "humanoid" if env_name == 'humanoid_simplfied' else env_name

def get_xml_string(env_name):
    """Get XML string from Brax environment assets."""
    env_name_for_brax = get_env_name_for_brax(env_name)
    xml_path = epath.resource_path('brax') / f'envs/assets/{env_name_for_brax}.xml'
    return xml_path.read_text()

def get_train_fn(env_name):
    """Get training function with pre-configured hyperparameters."""
    import functools
    from braxviewer.brax.training.agents.ppo import train as ppo
    from brax.training.agents.sac import train as sac
    
    train_fns = {
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
    }
    
    return train_fns[env_name]

def main():
    """Starts the WebViewer server."""
    # Get XML string for viewer
    xml_string = get_xml_string(config.ENV_NAME)
    
    # Get training function to determine num_envs
    train_fn = get_train_fn(config.ENV_NAME)
    num_envs = train_fn.keywords.get('num_envs', 8)  # default fallback
    
    viewer = WebViewer(
        host=config.HOST,
        port=config.PORT,
        xml=xml_string,
        num_envs=num_envs
    )
    # Run the server in a thread so it doesn't block.
    viewer.run()

    print(f"Viewer server started. Visit http://{config.HOST}:{config.PORT}")
    print(f"Environment: {config.ENV_NAME} (Brax: {get_env_name_for_brax(config.ENV_NAME)})")
    print("Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive to allow the server to run.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping viewer server...")
        viewer.stop()
        print("Server stopped.")

if __name__ == '__main__':
    main()