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
from braxviewer.utils import concatenate_envs_xml
# Assuming the custom ppo train function is located here
from braxviewer.brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


# ==============================================================================
#  Environment Definition
# ==============================================================================

xml_model = """
<mujoco model="inverted pendulum">
    <compiler inertiafromgeom="true"/>
    <default><joint armature="0" damping="1"/><geom contype="0" conaffinity="0" friction="1 0.1 0.1"/></default>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom name="rail" type="capsule"  size="0.02 1.5" pos="0 0 0" quat="1 0 1 0" rgba="1 1 1 1"/>
        <body name="cart" pos="0 0 0">
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
    
    # Set the number of parallel environments based on your code
    num_parallel_envs = 8

    # a) The original, single-agent env for training and evaluation.
    env_for_training = CartPole(xml_model=xml_model, backend='mjx')

    # b) The concatenated env for initializing the visualization scene.
    concatenated_xml = concatenate_envs_xml(
        xml_string=xml_model, num_envs=num_parallel_envs, env_separation=4.0, envs_per_row=4, base_body_name="cart"
    )
    env_for_visualization_init = CartPole(xml_model=concatenated_xml, backend='mjx')

    # Instantiate and run the viewer
    viewer = WebViewerBatched()
    viewer.run()  # Start the web server in the background

    # Initialize the viewer with the concatenated scene.
    viewer.init(env_for_visualization_init)

    # --- 2. Training Process ---

    # Define the PPO networks
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64, 64, 64, 64),
    )

    # Define the training function with parameters from your code.
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
        viewer=viewer,  # Hook the viewer into the training loop
    )

    # Define a progress callback function
    def progress_fn(current_step, metrics):
        if current_step > 0:
            print(f'Training Step: {current_step} \t Eval Reward: {metrics["eval/episode_reward"]:.3f}')

    # Execute training.
    make_policy_fn, params, _ = train_fn(
        environment=env_for_training,
        eval_env=env_for_training,
        progress_fn=progress_fn,
    )

    print("--- Training Complete ---")

    # Keep the script running after training.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.stop()
        print("Program stopped.")