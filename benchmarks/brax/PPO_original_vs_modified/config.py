
import jax
from brax.training.agents.ppo import networks as ppo_networks

# Configuration
HOST = "0.0.0.0"
BASE_PORT = 8090
NUM_ENVS = 8
NUM_BENCHMARK_RUNS = 5 # Number of runs to average, excluding the first warmup run

XML_MODEL = """
<mujoco model="inverted pendulum">
    <compiler inertiafromgeom="true"/>
    <default><joint armature="0" damping="1"/><geom contype="0" conaffinity="0" friction="1 0.1 0.1"/></default>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <body name="cart" pos="0 0 0">
            <geom name="rail" type="capsule"  size="0.02 1.5" pos="0 0 0" quat="1 0 1 0" rgba="1 1 1 1"/>
            <joint name="slider" type="slide" axis="1 0 0" pos="0 0 0" limited="true" range="-1.5 1.5"/>
            <geom name="cart_geom" pos="0 0 0" quat="1 0 1 0" size="0.1 0.1" type="capsule"/>
            <body name="pole" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom name="pole_geom" type="capsule" size="0.049 0.3"  fromto="0 0 0 0.001 0 0.6"/>
            </body>
        </body>
    </worldbody>
    <actuator><motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/></actuator>
</mujoco>
"""

HYPERPARAMS = {
    'small': {
        'num_timesteps': 10000,
        'num_evals': 5,
        'episode_length': 1000,
        'num_envs': 16,
        'batch_size': 8,
        'num_minibatches': 2,
        'unroll_length': 10,
        'num_updates_per_batch': 2,
    },
    'medium': {
        'num_timesteps': 50000,
        'num_evals': 10,
        'episode_length': 1000,
        'num_envs': 32,
        'batch_size': 16,
        'num_minibatches': 4,
        'unroll_length': 20,
        'num_updates_per_batch': 4,
    },
    'large': {
        'num_timesteps': 100000,
        'num_evals': 20,
        'episode_length': 1000,
        'num_envs': 64,
        'batch_size': 32,
        'num_minibatches': 8,
        'unroll_length': 30,
        'num_updates_per_batch': 8,
    }
}

COMMON_PARAMS = {
    'normalize_observations': True,
    'discounting': 0.97,
    'learning_rate': 3.0e-4,
    'entropy_cost': 1e-2,
    'network_factory': ppo_networks.make_ppo_networks,
    'seed': 0,
}
