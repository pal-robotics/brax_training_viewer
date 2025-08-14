import jax
import jax.numpy as jnp
import xml.etree.ElementTree as ET
from brax.envs.base import PipelineEnv, State

from braxviewer.WebViewer import WebViewer

class WebViewerBatched(WebViewer):
    def __init__(self,
                 grid_dims: tuple,
                 env_offset: tuple,
                 **kwargs):
        super().__init__(**kwargs)
        self.grid_dims = grid_dims
        self.env_offset = env_offset
        self.streamer.unbatched = False

    def init(self, env_concatenated: PipelineEnv):
        self.env = env_concatenated
        from jax import random as jax_random
        from brax.io import json
        rng = jax_random.PRNGKey(0)
        pipeline_state_to_serialize = None
        try:
            env_state = self.env.reset(rng)
            pipeline_state_to_serialize = env_state.pipeline_state
        except Exception:
            q = jnp.zeros((self.env.sys.q_size(),))
            qd = jnp.zeros((self.env.sys.qd_size(),))
            pipeline_state_to_serialize = self.env.pipeline_init(q, qd)
        sys_with_dt = self.env.sys.tree_replace({'opt.timestep': self.env.dt})
        self.system_json = json.dumps(sys_with_dt, [pipeline_state_to_serialize])
        self.send_control({"type": "refresh_web"})

    def send_frame(self, state):
        stitched = self.stitch_state(state)
        self.streamer.send(stitched, discard_queue=True)

    def stitch_state(self, batched_state: State) -> State:
        if not hasattr(batched_state, 'pipeline_state') or batched_state.pipeline_state.q.ndim < 2:
            return batched_state
        p = batched_state.pipeline_state
        num_envs = p.q.shape[0]

        cols, rows, _ = self.grid_dims
        envs_per_layer = cols * rows
        
        layer_indices = jnp.arange(num_envs) // envs_per_layer
        idx_in_layer = jnp.arange(num_envs) % envs_per_layer
        row_indices = idx_in_layer // cols
        col_indices = idx_in_layer % cols

        offsets_x = col_indices * self.env_offset[0]
        offsets_y = row_indices * self.env_offset[1]
        offsets_z = layer_indices * self.env_offset[2]
        
        world_offsets = jnp.stack([offsets_x, offsets_y, offsets_z], axis=1)[:, jnp.newaxis, :]
        offset_pos = p.x.pos + world_offsets
        stitched_q = jnp.reshape(p.q, (-1,))
        stitched_qd = jnp.reshape(p.qd, (-1,))
        stitched_x = p.x.replace(
            pos=jnp.reshape(offset_pos, (-1, 3)),
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
        return batched_state.replace(pipeline_state=stitched_pipeline_state)

    @staticmethod
    def concatenate_envs_xml(xml_string: str, num_envs: int, grid_dims: tuple, env_offset: tuple) -> str:
        """
        Duplicates all top-level elements from <worldbody> for each environment,
        and applies the grid offset to each duplicated element's position.
        This ensures correct physical behavior where static geoms remain static.
        """
        tree = ET.ElementTree(ET.fromstring(xml_string))
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("No <worldbody> found in the XML.")

        # Capture all original elements in the worldbody.
        base_elements = list(worldbody)
        
        cols, rows, _ = grid_dims
        envs_per_layer = cols * rows

        # Start loop from 1, as env 0 is the original and already in place.
        for i in range(1, num_envs):
            layer = i // envs_per_layer
            idx_in_layer = i % envs_per_layer
            row = idx_in_layer // cols
            col = idx_in_layer % cols
            
            x_offset = col * env_offset[0]
            y_offset = row * env_offset[1]
            z_offset = layer * env_offset[2]

            # For each original element, create a copy, offset its position, and add it to the world.
            for base_elem in base_elements:
                new_elem = ET.fromstring(ET.tostring(base_elem))

                current_pos_str = new_elem.get("pos", "0 0 0")
                current_pos = [float(p) for p in current_pos_str.split()]
                new_pos = [
                    current_pos[0] + x_offset,
                    current_pos[1] + y_offset,
                    current_pos[2] + z_offset,
                ]
                new_elem.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")

                # Suffix all names within the new element to avoid conflicts.
                for sub_elem in new_elem.iter():
                    if "name" in sub_elem.attrib:
                        sub_elem.set("name", f"{sub_elem.attrib['name']}_{i}")
                    if "joint" in sub_elem.attrib:
                        sub_elem.set("joint", f"{sub_elem.attrib['joint']}_{i}")
                
                worldbody.append(new_elem)

        # Handle actuators separately.
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