# utils.py (Optimized Version)

from brax.envs.base import State
from brax.base import State as BaseState
from jax import tree_util
import jax.numpy as jnp
import xml.etree.ElementTree as ET


def _physics_state_to_dict(phys_state: BaseState) -> dict:
    """Helper to convert a single, non-batched physics state to a dict."""
    out = {}
    # Directly convert arrays to lists.
    if hasattr(phys_state, 'q'):
        out['q'] = phys_state.q.tolist()
    if hasattr(phys_state, 'qd'):
        out['qd'] = phys_state.qd.tolist()

    if hasattr(phys_state, 'x'):
        out['x'] = {
            'pos': phys_state.x.pos.tolist(),
            'rot': phys_state.x.rot.tolist(),
        }

    if hasattr(phys_state, 'xd'):
        out['xd'] = {
            'vel': phys_state.xd.vel.tolist(),
            'ang': phys_state.xd.ang.tolist(),
        }
    
    # Contact info is optional and may not exist in all states.
    if hasattr(phys_state, 'contact'):
        contact = phys_state.contact
        contact_dict = {}
        # We only serialize link_idx for visualization purposes.
        if hasattr(contact, 'link_idx'):
            val = contact.link_idx
            # Handle tuple of arrays for link indices.
            if isinstance(val, tuple):
                contact_dict['link_idx'] = [v.tolist() for v in val]
            else:
                contact_dict['link_idx'] = val.tolist()
        if contact_dict:
            out['contact'] = contact_dict

    return out


def state_to_dict(state: State, index: int = 0, unbatched: bool = True) -> dict:
    """
    Converts a Brax State (which may be batched and nested) to a JSON-serializable dictionary.
    """
    # 1. First, get the actual physics state, which might be nested.
    if hasattr(state, 'pipeline_state') and state.pipeline_state is not None:
        phys_state = state.pipeline_state
    elif isinstance(state, BaseState):
        phys_state = state
    else:
        # If no valid physics state can be found, return an empty dict.
        return {}

    # 2. If unbatching is required, slice the physics state to get the desired frame.
    if unbatched:
        phys_state = tree_util.tree_map(
            lambda x: x[index] if isinstance(x, jnp.ndarray) and x.ndim > 0 and x.shape[0] > index else x,
            phys_state
        )

    # 3. Convert the final, single-frame physics state to a dictionary.
    return _physics_state_to_dict(phys_state)


def concatenate_envs_xml(xml_string: str, num_envs: int, env_separation: float, envs_per_row: int, base_body_name: str = None) -> str:
    """Concatenates multiple copies of a single-environment MuJoCo XML."""
    tree = ET.ElementTree(ET.fromstring(xml_string))
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in the XML.")

    if base_body_name:
        base_robot = worldbody.find(f"body[@name='{base_body_name}']")
    else:
        base_robot = worldbody.find("body")
    if base_robot is None:
        raise ValueError("No robot body found in the XML.")

    for i in range(1, num_envs):
        row = i // envs_per_row
        col = i % envs_per_row
        x_pos = col * env_separation
        y_pos = row * env_separation

        new_robot = ET.fromstring(ET.tostring(base_robot))
        new_robot.set("name", f"robot_{i}")
        new_robot.set("pos", f"{x_pos} {y_pos} 0")

        for elem in new_robot.iter():
            if "name" in elem.attrib:
                elem.set("name", f"{elem.attrib['name']}_{i}")
            if "joint" in elem.attrib:
                elem.set("joint", f"{elem.attrib['joint']}_{i}")

        worldbody.append(new_robot)

    actuator_root = root.find("actuator")
    if actuator_root is not None:
        base_actuators = list(actuator_root)
        for i in range(1, num_envs):
            for base_actuator in base_actuators:
                new_actuator = ET.fromstring(ET.tostring(base_actuator))
                if "name" in new_actuator.attrib:
                    new_actuator.set("name", f"{new_actuator.attrib['name']}_{i}")
                if "joint" in new_actuator.attrib:
                    new_actuator.set("joint", f"{new_actuator.attrib['joint']}_{i}")
                actuator_root.append(new_actuator)

    return ET.tostring(root, encoding="unicode")


def stitch_state(batched_state: State) -> State:
    """
    Stitches a batched state from parallel envs into a single large state
    for a concatenated environment.
    """
    if not hasattr(batched_state, 'pipeline_state') or batched_state.pipeline_state.q.ndim < 2:
        return batched_state

    p = batched_state.pipeline_state
    
    stitched_q = jnp.reshape(p.q, (-1,))
    stitched_qd = jnp.reshape(p.qd, (-1,))
    
    stitched_x = p.x.replace(
        pos=jnp.reshape(p.x.pos, (-1, 3)),
        rot=jnp.reshape(p.x.rot, (-1, 4))
    )
    
    stitched_xd = p.xd.replace(
        vel=jnp.reshape(p.xd.vel, (-1, 3)),
        ang=jnp.reshape(p.xd.ang, (-1, 3))
    )

    stitched_pipeline_state = p.replace(
        q=stitched_q,
        qd=stitched_qd,
        x=stitched_x,
        xd=stitched_xd
    )
    
    stitched_full_state = batched_state.replace(
        pipeline_state=stitched_pipeline_state
    )

    return stitched_full_state

# The unbatch_state function is no longer used by state_to_dict,
# but we leave it here in case it's needed elsewhere.
def unbatch_state(state: State, index: int) -> State:
    """Extracts the index-th sample from a batched brax State."""
    if not hasattr(state, 'pipeline_state'):
        return tree_util.tree_map(lambda x: x[index] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x, state)

    unbatched_pipeline = tree_util.tree_map(
        lambda x: x[index] if isinstance(x, jnp.ndarray) and x.ndim > 1 else x,
        state.pipeline_state
    )
    return state.replace(pipeline_state=unbatched_pipeline)