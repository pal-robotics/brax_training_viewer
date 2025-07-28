"""
Brax Training Speed Analysis Script
==================================

This script benchmarks the training speed of Brax PPO under different rendering conditions.
It measures the time for three scenarios:
  1. Viewer created and rendering ON (render_fn + render=True)
  2. Viewer created but rendering OFF (render_fn + render=False, avoids io_callback)
  3. No viewer (render_fn=None + render=False, pure Brax training)
"""

import time
import jax
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
import jax.numpy as jnp
from braxviewer.WebViewerParallel import WebViewerParallel
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.envs.wrappers.viewer import ViewerWrapper

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
    theta_init = jax.random.uniform(
        theta_key, (1,), minval=-0.1, maxval=0.1)[0]
    q_init = jnp.array([0.0, theta_init])
    qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)
    pipeline_state = self.pipeline_init(q_init, qd_init)
    reward, done = jnp.zeros(2)
    observation = self.get_observation(pipeline_state)
    metrics = {'rewards': reward}
    return State(
        pipeline_state=pipeline_state,
        obs=observation,
        reward=reward,
        done=done,
        metrics=metrics)

  def step(self, state: State, action: jax.Array) -> State:
    pipeline_state = self.pipeline_step(state.pipeline_state, action)
    observation = self.get_observation(pipeline_state)
    x, th = pipeline_state.q
    outside_x = jnp.abs(x) > 1.0
    outside_th = jnp.abs(th) > jnp.pi / 2
    done = jnp.float32(outside_x | outside_th)
    reward = jnp.cos(th)
    state.metrics.update({'rewards': reward})
    return state.replace(
        pipeline_state=pipeline_state, obs=observation, reward=reward, done=done)

  def get_observation(self, pipeline_state: State) -> jnp.ndarray:
    return jnp.concatenate([pipeline_state.q, pipeline_state.qd])


def main():
  make_networks_factory = ppo_networks.make_ppo_networks
  num_parallel_envs = 8

  times = {}
  base_port = 8000
  for i, render in enumerate([None, False, True]):
    port = base_port + i
    # A viewer is always created to ensure a logger is available.
    viewer = WebViewerParallel(
        num_envs=num_parallel_envs, xml=xml_model, port=port)

    if render is None:
      # Scenario 3: No viewer
      render_fn = None
      should_render = False
      env_for_training = CartPole(xml_model)
      viewer.log("--- Running benchmark: No viewer ---")
    else:
      # Scenarios 1 & 2: Viewer created
      viewer.run()
      env = CartPole(xml_model)
      env_for_training = ViewerWrapper(env, viewer)
      
      # Explicitly set the viewer's rendering state for the benchmark.
      # This ensures the dynamic check inside ppo.train works as expected.
      viewer.rendering_enabled = render

      if render:
        viewer.log(f"--- Running benchmark: Viewer ON (port {port}) ---")
      else:
        viewer.log(f"--- Running benchmark: Viewer OFF (port {port}) ---")

    start_time = time.time()
    _, _, _ = ppo.train(
        environment=env_for_training,
        num_timesteps=2000,
        num_evals=3,
        episode_length=300,
        num_envs=num_parallel_envs,
        network_factory=make_networks_factory,
        seed=0,
    )
    end_time = time.time()

    viewer.stop()

    times[str(render)] = end_time - start_time
    viewer.log(
        f"Finished run with render={render} in {times[str(render)]:.2f}s\n")

  viewer.log("\n--- Benchmark Results ---")
  for render_option, t in times.items():
    viewer.log(f"Render option {render_option}: {t:.2f}s")


if __name__ == '__main__':
  main()

