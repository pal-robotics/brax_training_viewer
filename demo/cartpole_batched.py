import sys
import os
import functools
import time

# Add project path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from braxviewer.brax.brax.io import mjcf
from braxviewer.brax.brax.envs.base import PipelineEnv, State

# Import custom modules and brax training modules
from braxviewer.WebViewerBatched import WebViewerBatched
from braxviewer.brax.brax.training.agents.ppo import train as ppo
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
    
    # grid_dims is now optional! If not provided, it will be auto-calculated
    # as a square grid: ceil(sqrt(num_envs)) x ceil(sqrt(num_envs)) x 1
    # For 8 envs: sqrt(8) = 2.83 -> ceil(2.83) = 3 -> (3, 3, 1) grid
    
    # If you want to override the auto-calculation, uncomment the lines below:
    # grid_dims = (4, 2, 1)  # For a flat 4x2 grid
    # grid_dims = (1, 1, 8)  # For a vertical stack
    
    # env_offset will be automatically calculated based on XML bounding box
    # If you want to override the auto-calculation, uncomment the line below:
    # env_offset_3d = (4.0, 4.0, 2.0)

    env_for_evaluation = CartPole(xml_model=xml_model, backend='mjx')

    # Instantiate the viewer with automatic XML concatenation
    # Both grid_dims and env_offset are now optional and will be auto-calculated
    # 
    # Default logging levels (you can override these if needed):
    # - log_level="info": Shows useful information like auto-calculated grid_dims and env_offset
    # - server_log_level="warning": Shows only important server events, not every HTTP request
    #
    # To customize logging, uncomment and modify:
    # log_level="debug"        # For more detailed application logs
    # server_log_level="info"  # For detailed server logs (HTTP requests, WebSocket connections)
    viewer = WebViewerBatched(
        # grid_dims=grid_dims,  # Optional: will auto-calculate if not provided
        # env_offset=env_offset_3d,  # Optional: will auto-calculate if not provided
        num_envs=num_parallel_envs,
        original_xml=xml_model,
        port=8080,
        host='127.0.0.1',
        # log_level="info",        # Default: Application logs (auto-calculation, etc.)
        # server_log_level="warning"  # Default: Server logs (only warnings and errors)
    )
    viewer.run()

    # Initialize viewer with concatenated XML (automatic)
    viewer.init()

    env_for_training = ViewerWrapper(env=env_for_evaluation, viewer=viewer)


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
        eval_env=env_for_evaluation,
        progress_fn=progress_fn,
    )

    print("--- Training Complete ---")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.stop()
        print("Program stopped.")