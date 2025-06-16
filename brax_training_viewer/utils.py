from brax import State
from jax import tree_util
import jax.numpy as jnp

def unbatch_state(state: State, index: int) -> State:
    """Extract the index-th sample from a batched brax State"""
    return tree_util.tree_map(lambda x: x[index] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x, state)

def state_to_dict(state: State, index: int = 0):
    """Convert a brax State (batched or unbatched) to a dict."""
 
    state = unbatch_state(state, index)

    out = {}

    if hasattr(state, 'q') and state.q is not None:
        out['q'] = state.q.tolist()
    if hasattr(state, 'qd') and state.qd is not None:
        out['qd'] = state.qd.tolist()

    if hasattr(state, 'x') and state.x is not None:
        x = state.x
        out['x'] = {}
        if hasattr(x, 'pos') and x.pos is not None:
            out['x']['pos'] = x.pos.tolist()
        if hasattr(x, 'rot') and x.rot is not None:
            out['x']['rot'] = x.rot.tolist()

    if hasattr(state, 'xd') and state.xd is not None:
        xd = state.xd
        out['xd'] = {}
        if hasattr(xd, 'vel') and xd.vel is not None:
            out['xd']['vel'] = xd.vel.tolist()
        if hasattr(xd, 'ang') and xd.ang is not None:
            out['xd']['ang'] = xd.ang.tolist()

    if hasattr(state, 'contact') and state.contact is not None:
        contact = state.contact
        contact_dict = {}
        for attr in ['link_idx', 'elasticity']:
            if hasattr(contact, attr):
                val = getattr(contact, attr)
                # Handle tuple of arrays (e.g., link_idx = (array, array))
                if isinstance(val, tuple):
                    contact_dict[attr] = [v.tolist() for v in val]
                else:
                    contact_dict[attr] = val.tolist()
        if contact_dict:
            out['contact'] = contact_dict

    return out
