#!/usr/bin/env python3
"""
Simple Brax Training Speed Benchmark
====================================

Run this script to benchmark Brax PPO training under 3 scenarios:
1. No viewer (pure training)
2. Viewer OFF (viewer created but rendering disabled)
3. Viewer ON (full rendering enabled)
"""

import time
import requests
import jax
import jax.numpy as jnp
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
from braxviewer.WebViewer import WebViewer
from braxviewer.BraxSender import BraxSender
from braxviewer.brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from braxviewer.wrapper import ViewerWrapper

# Configuration
HOST = "0.0.0.0"
BASE_PORT = 8090
NUM_ENVS = 8

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

class CartPole(PipelineEnv):
    def __init__(self, backend: str = 'mjx', **kwargs):
        sys = mjcf.loads(XML_MODEL)
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

def wait_for_rendering_status(host, port, expected_status, timeout=10):
    """Wait for rendering status to match expected value."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            url = f"http://{host}:{port}/api/rendering_status"
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                current_status = response.json().get("rendering_enabled", True)
                if current_status == expected_status:
                    print(f"Rendering status confirmed: {current_status}")
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    print(f"Timeout waiting for rendering status to become {expected_status}")
    return False

def main():
    make_networks_factory = ppo_networks.make_ppo_networks
    times = {}
    
    scenarios = [
        (None, "No viewer"),
        (False, "Viewer OFF"), 
        (True, "Viewer ON")
    ]
    
    print("Running Brax training speed benchmark...")
    
    for i, (render, scenario_name) in enumerate(scenarios):
        port = BASE_PORT + i
        print(f"\nTesting: {scenario_name}")
        
        if render is None:
            env_for_training = CartPole()
        else:
            viewer = WebViewer(host=HOST, port=port, xml=XML_MODEL, num_envs=NUM_ENVS)
            viewer.run()
            
            sender = BraxSender(host=HOST, port=port, xml=XML_MODEL, num_envs=NUM_ENVS)
            sender.start()
            
            env = CartPole()
            env_for_training = ViewerWrapper(env=env, sender=sender)
            
            # Set rendering status and wait for it to take effect
            if render is False:
                print("Disabling rendering...")
                sender.set_rendering_enabled(False)
                wait_for_rendering_status(HOST, port, False)
            elif render is True:
                print("Enabling rendering...")
                sender.set_rendering_enabled(True)
                wait_for_rendering_status(HOST, port, True)

        start_time = time.time()
        _, _, _ = ppo.train(
            environment=env_for_training,
            num_timesteps=2000,
            num_evals=3,
            episode_length=300,
            num_envs=NUM_ENVS,
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
        end_time = time.time()

        if render is not None:
            sender.stop()
            viewer.stop()

        elapsed_time = end_time - start_time
        times[scenario_name] = elapsed_time
        print(f"Time: {elapsed_time:.2f}s")

    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    for scenario, time_taken in times.items():
        print(f"{scenario}: {time_taken:.2f}s")

if __name__ == '__main__':
    main() 