# Shared configuration for the Brax Environments example.

# Network configuration
HOST = "127.0.0.1"
PORT = 8000

# Environment and training configuration
# This must be consistent between the viewer and the training script.
ENV_NAME = 'humanoid_simplfied'  # ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
BACKEND = 'mjx'  # ['generalized', 'positional', 'spring', 'mjx']

# Map simplified names to actual brax environment names
ENV_NAME_MAPPING = {
    'humanoid_simplfied': 'humanoid'
}

# Get the actual environment name for brax
ENV_NAME_FOR_BRAX = ENV_NAME_MAPPING.get(ENV_NAME, ENV_NAME)

# Training hyperparameters for different environments
TRAINING_CONFIGS = {
    'inverted_pendulum': {
        'num_timesteps': 2_000_000,
        'num_evals': 20,
        'reward_scaling': 10,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 5,
        'num_minibatches': 32,
        'num_updates_per_batch': 4,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-2,
        'num_envs': 2048,
        'batch_size': 1024,
        'seed': 1
    },
    'inverted_double_pendulum': {
        'num_timesteps': 20_000_000,
        'num_evals': 20,
        'reward_scaling': 10,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 5,
        'num_minibatches': 32,
        'num_updates_per_batch': 4,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-2,
        'num_envs': 2048,
        'batch_size': 1024,
        'seed': 1
    },
    'ant': {
        'num_timesteps': 50_000_000,
        'num_evals': 10,
        'reward_scaling': 10,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 5,
        'num_minibatches': 32,
        'num_updates_per_batch': 4,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-2,
        'num_envs': 4096,
        'batch_size': 2048,
        'seed': 1
    },
    'humanoid_simplfied': {
        'num_timesteps': 40000,
        'num_evals': 10,
        'reward_scaling': 0.1,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 10,
        'num_minibatches': 2,
        'num_updates_per_batch': 8,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-3,
        'num_envs': 4,
        'batch_size': 2,
        'seed': 1
    },
    'humanoid': {
        'num_timesteps': 50_000_000,
        'num_evals': 10,
        'reward_scaling': 0.1,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 10,
        'num_minibatches': 32,
        'num_updates_per_batch': 8,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-3,
        'num_envs': 2048,
        'batch_size': 1024,
        'seed': 1
    },
    'reacher': {
        'num_timesteps': 50_000_000,
        'num_evals': 20,
        'reward_scaling': 5,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 4,
        'unroll_length': 50,
        'num_minibatches': 32,
        'num_updates_per_batch': 8,
        'discounting': 0.95,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-3,
        'num_envs': 2048,
        'batch_size': 256,
        'max_devices_per_host': 8,
        'seed': 1
    },
    'humanoidstandup': {
        'num_timesteps': 100_000_000,
        'num_evals': 20,
        'reward_scaling': 0.1,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 15,
        'num_minibatches': 32,
        'num_updates_per_batch': 8,
        'discounting': 0.97,
        'learning_rate': 6e-4,
        'entropy_cost': 1e-2,
        'num_envs': 2048,
        'batch_size': 1024,
        'seed': 1
    },
    'hopper': {
        'num_timesteps': 6_553_600,
        'num_evals': 20,
        'reward_scaling': 30,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'discounting': 0.997,
        'learning_rate': 6e-4,
        'num_envs': 128,
        'batch_size': 512,
        'grad_updates_per_step': 64,
        'max_devices_per_host': 1,
        'max_replay_size': 1048576,
        'min_replay_size': 8192,
        'seed': 1
    },
    'walker2d': {
        'num_timesteps': 7_864_320,
        'num_evals': 20,
        'reward_scaling': 5,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'discounting': 0.997,
        'learning_rate': 6e-4,
        'num_envs': 128,
        'batch_size': 128,
        'grad_updates_per_step': 32,
        'max_devices_per_host': 1,
        'max_replay_size': 1048576,
        'min_replay_size': 8192,
        'seed': 1
    },
    'halfcheetah': {
        'num_timesteps': 50_000_000,
        'num_evals': 20,
        'reward_scaling': 1,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 20,
        'num_minibatches': 32,
        'num_updates_per_batch': 8,
        'discounting': 0.95,
        'learning_rate': 3e-4,
        'entropy_cost': 0.001,
        'num_envs': 2048,
        'batch_size': 512,
        'seed': 3
    },
    'pusher': {
        'num_timesteps': 50_000_000,
        'num_evals': 20,
        'reward_scaling': 5,
        'episode_length': 1000,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 30,
        'num_minibatches': 16,
        'num_updates_per_batch': 8,
        'discounting': 0.95,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-2,
        'num_envs': 2048,
        'batch_size': 512,
        'seed': 3
    }
}

# Get configuration for current environment
TRAINING_CONFIG = TRAINING_CONFIGS[ENV_NAME]
NUM_ENVS = TRAINING_CONFIG['num_envs'] 