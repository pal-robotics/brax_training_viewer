import sys, os
sys.path.insert(0, '/Users/shouyuehu/Projects/brax_training_viewer/brax_training_viewer')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brax
print("Using brax from:", brax.__file__)

# JAX Imports:
import jax
import jax.numpy as jnp

# Brax Imports:
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State

import functools

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo


from brax_training_viewer.websocket_streamer import WebSocketStateStreamer
from brax_training_viewer.brax_ow.io import html

xml_model = """
<mujoco model="inverted pendulum">
    <compiler inertiafromgeom="true"/>

    <default>
        <joint armature="0" damping="1"/>
        <geom contype="0" conaffinity="0" friction="1 0.1 0.1"/>
    </default>

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

    <actuator>
        <motor ctrllimited="true" ctrlrange="-3 3" gear="100" joint="slider" name="slide"/>
    </actuator>

</mujoco>
"""

# # Load the MJCF model
# sys = mjcf.loads(xml_model)

# # Jitting the init and step functions for GPU acceleration
# init_fn = jax.jit(pipeline.init)
# step_fn = jax.jit(pipeline.step)

# # Initializing the state:
# state = init_fn(
#     sys=sys, q=sys.init_q, qd=jnp.zeros(sys.qd_size()),
# )

# # Running the simulation
# num_steps = 1000
# num_steps = 1
# state_history = []
# ctrl = jnp.zeros(sys.act_size())
# for i in range(num_steps):
#     state = step_fn(sys, state, act=ctrl)
#     state_history.append(state)


# # MJX Backend Requires Contact Information even if it does not exist:
# state_history = list(map(lambda x: x.replace(contact=None), state_history))

# Environment:
class CartPole(PipelineEnv):
    """ Environment for training Cart Pole balancing """

    def __init__(self, xml_model: str, backend: str = 'mjx', **kwargs):
        # Initialize System:
        sys = mjcf.loads(xml_model)
        self.step_dt = 0.02
        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend=backend, n_frames=n_frames)

    def reset(self, rng: jax.Array) -> State:
        key, theta_key, qd_key = jax.random.split(rng, 3)

        theta_init = jax.random.uniform(theta_key, (1,), minval=-0.1, maxval=0.1)[0]
        
        # q structure: [x th]
        q_init = jnp.array([0.0, theta_init])
        
        # qd structure: [dx dth]
        qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)        
        
        # Initialize State:
        pipeline_state = self.pipeline_init(q_init, qd_init)

        # Initialize Rewards:
        reward, done = jnp.zeros(2)

        # Get observation for RL Algorithm (Input to our neural net):
        observation = self.get_observation(pipeline_state)

        # Metrics:
        metrics = {
            'rewards': reward,
            # 'observation': observation,
        }

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        # Forward Physics Step:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Get Observation from new state:
        observation = self.get_observation(pipeline_state)

        # Extract States:
        x, th = pipeline_state.q

        # Terminate if outside the range of the rail and pendulum is past the rail:
        outside_x = jnp.abs(x) > 1.0
        outside_th = jnp.abs(th) > jnp.pi / 2
        done = outside_x | outside_th
        done = jnp.float32(done)

        # Calculate Reward:
        reward = jnp.cos(th)

        metrics = {
            'rewards': reward,
            # 'observation': observation,
        }
        state.metrics.update(metrics)

        # Update State object:
        state = state.replace(
            pipeline_state=pipeline_state, obs=observation, reward=reward, done=done,
        )
        return state

    def get_observation(self, pipeline_state: State) -> jnp.ndarray:
        # Observation: [x, th, dx, dth]
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])
    





def main():
    # Create the streamer object (once)
    streamer = WebSocketStateStreamer()
    streamer.start()

    env = CartPole(xml_model=xml_model, backend='mjx')
    eval_env = CartPole(xml_model=xml_model, backend='mjx')

    def progress_fn(current_step, metrics):
        if current_step > 0:
            print(f'Step: {current_step} \t Reward: Episode Reward: {metrics["eval/episode_reward"]:.3f}')
            eval_key = jax.random.PRNGKey(current_step)



    #TODO streamer.send(state.pipeline_state)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64, 64, 64, 64),
    )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=40000,
        num_evals=10,
        episode_length=300,
        num_envs=4,
        num_eval_envs=4,
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

    # Train the policy
    make_policy_fn, params, _ = train_fn(
    environment=env,
    progress_fn=progress_fn,
    eval_env=eval_env,
    )


if __name__ == "__main__":
    main()


