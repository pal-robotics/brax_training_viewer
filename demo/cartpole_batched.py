import sys
import os
import functools
import time

# Add project path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State

# Import custom modules and brax training modules
from braxviewer.WebViewerBatched import WebViewerBatched
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from braxviewer.brax.brax.envs.wrappers.viewer import ViewerWrapper

# ==============================================================================
#  Environment Definition
# ==============================================================================

xml_model = """
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

class CartPole(PipelineEnv):
    def __init__(self, xml_model: str, backend: str = 'mjx', **kwargs):
        sys = mjcf.loads(xml_model)
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
#  Main Execution Logic
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Environment and Viewer Setup ---
    
    num_parallel_envs = 8
    
    # Define the 3D grid layout: (cols, rows, layers)
    # For a 2x2x2 cube:
    grid_dims = (4, 2, 1)
    # For a flat 4x2 grid, use: grid_dims = (4, 2, 1)
    # For a vertical stack, use: grid_dims = (1, 1, 8)
    
    # Define the spacing for each grid cell
    env_offset_3d = (4.0, 4.0, 2.0)

    # Ensure the grid dimensions can hold all environments
    grid_capacity = grid_dims[0] * grid_dims[1] * grid_dims[2]
    assert num_parallel_envs <= grid_capacity, f"Grid dimensions {grid_dims} can only hold {grid_capacity} envs, but {num_parallel_envs} were requested."

    env_for_training = CartPole(xml_model=xml_model, backend='mjx')

    concatenated_xml = WebViewerBatched.concatenate_envs_xml(
        xml_string=xml_model, 
        num_envs=num_parallel_envs, 
        grid_dims=grid_dims, 
        env_offset=env_offset_3d
    )
    env_for_visualization_init = CartPole(xml_model=concatenated_xml, backend='mjx')

    # Instantiate the viewer, passing 3D layout info to the constructor.
    viewer = WebViewerBatched(
        grid_dims=grid_dims,
        env_offset=env_offset_3d
    )
    viewer.run()

    viewer.init(env_for_visualization_init)

    env_for_training = ViewerWrapper(env=env_for_training, viewer=viewer)


    # --- 2. Training Process ---

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64, 64, 64, 64),
    )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=40000,
        num_evals=10,
        episode_length=300,
        num_envs=num_parallel_envs,
        num_eval_envs=num_parallel_envs,
        batch_size=4,
        num_minibatches=4,
        unroll_length=20,
        num_updates_per_batch=4,
        normalize_observations=True,
        discounting=0.97,
        learning_rate=3.0e-4,
        entropy_cost=1e-2,
        network_factory=make_networks_factory,
        seed=0,
    )

    def progress_fn(current_step, metrics):
        if current_step > 0:
            print(f'Training Step: {current_step} \t Eval Reward: {metrics["eval/episode_reward"]:.3f}')

    make_policy_fn, params, _ = train_fn(
        environment=env_for_training,
        eval_env=env_for_training,
        progress_fn=progress_fn,
    )

    print("--- Training Complete ---")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.stop()
        print("Program stopped.")