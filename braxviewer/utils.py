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