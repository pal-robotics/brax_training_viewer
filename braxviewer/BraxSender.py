import jax.numpy as jnp
import xml.etree.ElementTree as ET
import json as std_json
import math
import requests
import threading
import time
from brax.envs.base import State

from braxviewer.core.config import Config
from braxviewer.core.Sender import Sender

class BraxSender(Sender):
    def __init__(self, config=None, **kwargs):
        if config is None:
            from braxviewer.core.config import Config
            config = Config(**kwargs)
        super().__init__(config)
        self.config = config # Ensure type hint is for Config
        
        if self.config.grid_dims is None:
            if self.config.num_envs is None:
                raise ValueError("num_envs must be provided when grid_dims is not specified")
            self.config.grid_dims = self._auto_calculate_grid_dims(self.config.num_envs)
            self.log(f"Auto-calculated grid_dims: {self.config.grid_dims}")

        if self.config.env_offset is None:
            self.config.env_offset = self._auto_calculate_env_offset(self.config.xml)
            self.log(f"Auto-calculated env_offset: {self.config.env_offset}")

        self.concatenated_xml = self._concatenate_envs_xml()
        
        # Start status polling thread
        self._status_thread = None
        self._stop_polling = False

    def _poll_rendering_status(self):
        """Poll the server for rendering status updates."""
        last_status = None
        consecutive_failures = 0
        
        while not self._stop_polling:
            try:
                url = f"http://{self.config.host}:{self.config.port}/api/rendering_status"
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    data = response.json()
                    server_rendering_enabled = data.get("rendering_enabled", True)
                    
                    # Only log when status actually changes
                    if server_rendering_enabled != self.rendering_enabled:
                        self.rendering_enabled = server_rendering_enabled
                        self.log(f"Rendering status updated to: {self.rendering_enabled}")
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                else:
                    consecutive_failures += 1
                    
            except Exception as e:
                consecutive_failures += 1
                # Only log after multiple consecutive failures
                if consecutive_failures == 5:
                    self.log(f"Server connection issues (attempt {consecutive_failures})", level="warning")
                elif consecutive_failures == 20:
                    self.log(f"Server still unavailable after {consecutive_failures} attempts", level="warning")
            
            # Increase polling interval on failures
            if consecutive_failures > 10:
                time.sleep(5.0)  # 5 seconds when having issues
            else:
                time.sleep(1.0)  # 1 second normally  # Poll every 1 second

    def start(self):
        super().start()
        # Start status polling
        self._status_thread = threading.Thread(target=self._poll_rendering_status, daemon=True)
        self._status_thread.start()

    def stop(self):
        self._stop_polling = True
        super().stop()

    def set_rendering_enabled(self, enabled: bool):
        """Set rendering enabled state by updating the server via HTTP POST."""
        try:
            url = f"http://{self.config.host}:{self.config.port}/api/rendering_status"
            data = {"rendering_enabled": enabled}
            response = requests.post(url, json=data, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    self.rendering_enabled = enabled
                    self.log(f"Rendering status updated to: {enabled}")
                    return True
                else:
                    self.log(f"Server returned error: {result.get('message', 'Unknown error')}", level="error")
                    return False
            else:
                self.log(f"HTTP POST failed with status {response.status_code}: {response.text}", level="error")
                return False
        except requests.exceptions.ConnectionError:
            self.log("Cannot connect to server to update rendering status", level="error")
            return False
        except Exception as e:
            self.log(f"Failed to update rendering status: {e}", level="error")
            return False

    def init(self):
        # BraxGridSender has a slightly different init flow
        super().init(self.concatenated_xml)

    def send_frame(self, state: State, env_id: int = 0):
        if not self.rendering_enabled:
            return

        cols, rows, _ = self.config.grid_dims
        envs_per_layer = cols * rows
        
        layer = env_id // envs_per_layer
        idx_in_layer = env_id % envs_per_layer
        row = idx_in_layer // cols
        col = idx_in_layer % cols
        
        x_offset = col * self.config.env_offset[0]
        y_offset = row * self.config.env_offset[1]
        z_offset = layer * self.config.env_offset[2]
        
        pos_list = state.pipeline_state.x.pos.tolist()
        rot_list = state.pipeline_state.x.rot.tolist()
        
        offset_pos_list = [[p[0] + x_offset, p[1] + y_offset, p[2] + z_offset] for p in pos_list]

        frame = {
            'x': {'pos': offset_pos_list, 'rot': rot_list},
            'env_id': env_id,
        }
        
        self.streamer.send(std_json.dumps(frame), discard_queue=True)

    def stitch_state(self, batched_state: State) -> State:
        if not hasattr(batched_state, 'pipeline_state') or batched_state.pipeline_state.q.ndim < 2:
            return batched_state
        p = batched_state.pipeline_state
        num_envs = p.q.shape[0]

        cols, rows, _ = self.config.grid_dims
        envs_per_layer = cols * rows
        
        layer_indices = jnp.arange(num_envs) // envs_per_layer
        idx_in_layer = jnp.arange(num_envs) % envs_per_layer
        row_indices = idx_in_layer // cols
        col_indices = idx_in_layer % cols

        offsets_x = col_indices * self.config.env_offset[0]
        offsets_y = row_indices * self.config.env_offset[1]
        offsets_z = layer_indices * self.config.env_offset[2]
        
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
    
    def _auto_calculate_grid_dims(self, num_envs: int) -> tuple:
        if num_envs <= 0: return (1, 1, 1)
        grid_size = math.ceil(math.sqrt(num_envs))
        return (grid_size, grid_size, 1)

    def _auto_calculate_env_offset(self, xml_string: str) -> tuple:
        if not xml_string: return (4.0, 4.0, 2.0)
        try:
            bbox = self._calculate_xml_bounding_box(xml_string)
            padding_factor = 1.5
            x_size = (bbox['max_x'] - bbox['min_x']) * padding_factor
            y_size = (bbox['max_y'] - bbox['min_y']) * padding_factor
            z_size = (bbox['max_z'] - bbox['min_z']) * padding_factor
            min_spacing = 2.0
            return (max(x_size, min_spacing), max(y_size, min_spacing), max(z_size, min_spacing))
        except Exception as e:
            self.log(f"Warning: Failed to auto-calculate env_offset: {e}", level='warning')
            return (4.0, 4.0, 2.0)

    def _calculate_xml_bounding_box(self, xml_string: str) -> dict:
        tree = ET.ElementTree(ET.fromstring(xml_string))
        root = tree.getroot()
        
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
        max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

        def update_bounds(pos, size_info):
            nonlocal min_x, min_y, min_z, max_x, max_y, max_z
            x, y, z = pos
            dx, dy, dz = size_info
            min_x, max_x = min(min_x, x - dx), max(max_x, x + dx)
            min_y, max_y = min(min_y, y - dy), max(max_y, y + dy)
            min_z, max_z = min(min_z, z - dz), max(max_z, z + dz)

        def parse_position(pos_str):
            if not pos_str: return [0.0, 0.0, 0.0]
            return [float(x) for x in pos_str.split()]

        def calculate_geom_bounds(geom, parent_pos=[0, 0, 0]):
            if 'type' not in geom.attrib:
                name = geom.get('name', '[unnamed]')
                self.log(f"Warning: geom '{name}' is missing 'type' attribute. Skipping for bounding box calculation.", level='warning')
                return

            geom_type = geom.get('type')

            if geom_type == 'plane':
                return
            
            pos = parse_position(geom.get('pos', '0 0 0'))
            pos = [pos[i] + parent_pos[i] for i in range(3)]
            
            if geom_type == 'box':
                size = parse_position(geom.get('size', '0.1 0.1 0.1'))
                update_bounds(pos, size)
            elif geom_type == 'sphere':
                radius = float(geom.get('size', '0.1').split()[0])
                update_bounds(pos, [radius, radius, radius])
            elif geom_type == 'capsule':
                fromto = geom.get('fromto')
                if fromto:
                    coords = [float(x) for x in fromto.split()]
                    p1, p2 = coords[:3], coords[3:]
                    radius = float(geom.get('size', '0.1').split()[0])
                    update_bounds([p1[i] + parent_pos[i] for i in range(3)], [radius, radius, radius])
                    update_bounds([p2[i] + parent_pos[i] for i in range(3)], [radius, radius, radius])
                else:
                    size_parts = geom.get('size', '0.1 0.1').split()
                    radius = float(size_parts[0])
                    length = float(size_parts[1]) if len(size_parts) > 1 else radius
                    update_bounds(pos, [radius, radius, length])
            elif geom_type == 'cylinder':
                size_parts = geom.get('size', '0.1 0.1').split()
                radius = float(size_parts[0])
                height = float(size_parts[1]) if len(size_parts) > 1 else radius
                update_bounds(pos, [radius, radius, height])

        def traverse_body(body, parent_pos=[0, 0, 0]):
            body_pos = parse_position(body.get('pos', '0 0 0'))
            current_pos = [body_pos[i] + parent_pos[i] for i in range(3)]
            for geom in body.findall('geom'):
                calculate_geom_bounds(geom, current_pos)
            for child_body in body.findall('body'):
                traverse_body(child_body, current_pos)

        worldbody = root.find('worldbody')
        if worldbody is not None:
            for geom in worldbody.findall('geom'):
                calculate_geom_bounds(geom)
            for body in worldbody.findall('body'):
                traverse_body(body)

        if min_x == float('inf'):
            min_x = min_y = min_z = -1.0
            max_x = max_y = max_z = 1.0

        return {
            'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 
            'max_y': max_y, 'min_z': min_z, 'max_z': max_z
        }
    
    def _concatenate_envs_xml(self) -> str:
        tree = ET.ElementTree(ET.fromstring(self.config.xml))
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None: raise ValueError("No <worldbody> found in XML.")
        
        base_elements = list(worldbody)
        cols, rows, _ = self.config.grid_dims
        envs_per_layer = cols * rows

        for i in range(1, self.config.num_envs):
            layer = i // envs_per_layer
            idx_in_layer = i % envs_per_layer
            row = idx_in_layer // cols
            col = idx_in_layer % cols
            
            x_offset = col * self.config.env_offset[0]
            y_offset = row * self.config.env_offset[1]
            z_offset = layer * self.config.env_offset[2]

            for base_elem in base_elements:
                new_elem = ET.fromstring(ET.tostring(base_elem))
                pos_str = new_elem.get("pos", "0 0 0")
                pos = [float(p) for p in pos_str.split()]
                new_pos = [pos[0] + x_offset, pos[1] + y_offset, pos[2] + z_offset]
                new_elem.set("pos", " ".join(map(str, new_pos)))
                
                for sub_elem in new_elem.iter():
                    if "name" in sub_elem.attrib: sub_elem.set("name", f"{sub_elem.attrib['name']}_{i}")
                    if "joint" in sub_elem.attrib: sub_elem.set("joint", f"{sub_elem.attrib['joint']}_{i}")
                
                worldbody.append(new_elem)

        actuator_root = root.find("actuator")
        if actuator_root:
            base_actuators = list(actuator_root)
            for i in range(1, self.config.num_envs):
                for base_actuator in base_actuators:
                    new_actuator = ET.fromstring(ET.tostring(base_actuator))
                    if "name" in new_actuator.attrib: new_actuator.set("name", f"{new_actuator.attrib['name']}_{i}")
                    if "joint" in new_actuator.attrib: new_actuator.set("joint", f"{new_actuator.attrib['joint']}_{i}")
                    actuator_root.append(new_actuator)

        return ET.tostring(root, encoding="unicode") 