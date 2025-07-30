# Shared configuration for the Cartpole example.

# Network configuration
HOST = "127.0.0.1"
PORT = 8000

# Environment and training configuration
# This must be consistent between the viewer and the training script.
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

# --- PPO Hyperparameters ---
# See: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
NUM_PARALLEL_ENVS = 8
NUM_TIMESTEPS = 40000
NUM_EVALS = 10
EPISODE_LENGTH = 300
BATCH_SIZE = 4
NUM_MINIBATCHES = 4
UNROLL_LENGTH = 20
NUM_UPDATES_PER_BATCH = 4
NORMALIZE_OBSERVATIONS = True
DISCOUNTING = 0.97
LEARNING_RATE = 3.0e-4
ENTROPY_COST = 1e-2
SEED = 0 