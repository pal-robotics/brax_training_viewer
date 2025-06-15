from brax import State

def state_to_dict(state: State):
    """Convert a brax State to a dict, handling all data if exists."""
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
