import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GUI} from 'lilgui';

import {Animator} from './animator.js';
import {Selector} from './selector.js';
import {createScene, createTrajectory} from './system.js';

function downloadDataUri(name, uri) {
  let link = document.createElement('a');
  link.download = name;
  link.href = uri;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadFile(name, contents, mime) {
  mime = mime || 'text/plain';
  let blob = new Blob([contents], {type: mime});
  let link = document.createElement('a');
  document.body.appendChild(link);
  link.download = name;
  link.href = window.URL.createObjectURL(blob);
  link.onclick = function(e) {
    let scope = this;
    setTimeout(function() {
      window.URL.revokeObjectURL(scope.href);
    }, 1500);
  };
  link.click();
  link.remove();
}

function toggleDebugMode(obj, debug) {
  for (let i = 0; i < obj.children.length; i++) {
    let c = obj.children[i];
    if (c.type == 'AxesHelper') c.visible = debug;
    if (c.type == 'Group') {
      for (let j = 0; j < c.children.length; j++) {
        toggleDebugMode(c.children[j], debug);
      }
    }
  }
  if (obj.type == 'Mesh') {
    if (debug) {
      obj.material.opacity = 0.6;
      obj.material.transparent = true;
    } else {
      obj.material.opacity = 1.0;
      obj.material.transparent = false;
    }
  }
}

const hoverMaterial =
    new THREE.MeshPhongMaterial({color: 0x332722, emissive: 0x114a67});
const selectMaterial = new THREE.MeshPhongMaterial({color: 0x2194ce});

class Viewer {
  constructor(domElement, system) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

    this.domElement = domElement;
    this.system = system;
    this.scene = createScene(system);
    this.trajectory = createTrajectory(system);

    // For batched envs, calculate the number of links in a single sub-env.
    // This is used to map global link indices to local ones.
    this.numLinksPerEnv = 0;
    // A batched env is detected if any geom name ends with `_` followed by a digit.
    const isBatched = Object.keys(this.system.geoms).some(name => /_\d+$/.test(name));

    if (isBatched) {
        const env0_link_indices = new Set();
        // Geoms for env 0 are those that DO NOT end in `_` followed by a digit.
        for (const [name, geoms] of Object.entries(this.system.geoms)) {
            if (!/_\d+$/.test(name)) {
                const link_idx = geoms[0].link_idx;
                if (link_idx != null && link_idx >= 0) {
                    env0_link_indices.add(link_idx);
                }
            }
        }
        this.numLinksPerEnv = env0_link_indices.size;
    }

    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
    this.renderer.shadowMap.enabled = true;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.domElement.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    this.camera.position.set(5, 8, 2);
    this.camera.follow = true;
    this.camera.freezeAngle = false;
    this.camera.followDistance = 10;

    this.scene.background = new THREE.Color(0xa0a0a0);
    this.scene.fog = new THREE.Fog(0xa0a0a0, 40, 60);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
    hemiLight.position.set(0, 20, 0);
    this.scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff);
    dirLight.position.set(3, 10, 10);
    dirLight.castShadow = true;
    dirLight.shadow.camera.top = 10;
    dirLight.shadow.camera.bottom = -10;
    dirLight.shadow.camera.left = -10;
    dirLight.shadow.camera.right = 10;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 40;
    dirLight.shadow.mapSize.width = 4096;
    dirLight.shadow.mapSize.height = 4096;
    this.scene.add(dirLight);
    this.dirLight = dirLight;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.addEventListener('start', () => this.setDirty());
    this.controls.addEventListener('change', () => this.setDirty());
    this.controlTargetPos = this.controls.target.clone();

    this.gui = new GUI({autoPlace: false});
    this.domElement.parentElement.appendChild(this.gui.domElement);
    this.gui.domElement.style.position = 'absolute';
    this.gui.domElement.style.right = 0;
    this.gui.domElement.style.top = 0;
    
    // Add rendering toggle switch
    this.rendering = { enabled: true };
    const renderFolder = this.gui.addFolder('Real-time Rendering');
    renderFolder.add(this.rendering, 'enabled').name('Enable')
      .onChange((value) => {
        if (window.control_ws) {
            window.control_ws.send(JSON.stringify({ type: 'toggle_render', enabled: value }));
        }
      });
    renderFolder.open();

    const cameraFolder = this.gui.addFolder('Camera');
    cameraFolder.add(this.camera, 'freezeAngle').name('Freeze Angle');
    cameraFolder.add(this.camera, 'follow').name('Follow Target');
    cameraFolder.add(this.camera, 'followDistance').name('Follow Distance').min(1).max(50);

    this.animator = new Animator(this);
    this.animator.load(this.trajectory, {});

    const bodiesFolder = this.gui.addFolder('Bodies');
    bodiesFolder.close();
    this.bodyFolders = {};

    for (let c of this.scene.children) {
      if (!c.name) continue;
      const folder = bodiesFolder.addFolder(c.name);
      this.bodyFolders[c.name] = folder;
      folder.close();
      function defaults() {
        for (const gui of arguments) {
          gui.step(0.01).listen().domElement.style.pointerEvents = 'none';
        }
      }
      defaults(
        folder.add(c.position, 'x').name('pos.x'),
        folder.add(c.position, 'y').name('pos.y'),
        folder.add(c.position, 'z').name('pos.z'),
        folder.add(c.rotation, 'x').name('rot.x'),
        folder.add(c.rotation, 'y').name('rot.y'),
        folder.add(c.rotation, 'z').name('rot.z'),
      );
    }

    let saveFolder = this.gui.addFolder('Save / Capture');
    saveFolder.add(this, 'saveScene').name('Save Scene');
    saveFolder.add(this, 'saveImage').name('Capture Image');
    saveFolder.close();

    this.debugMode = false;
    let debugFolder = this.gui.addFolder('Debugger');
    debugFolder.add(this, 'debugMode')
        .name('axis')
        .onChange((value) => this.setDebugMode(value));
    this.gui.close();

    this.selector = new Selector(this);
    this.selector.addEventListener('hoveron', (evt) => this.setHover(evt.object, true));
    this.selector.addEventListener('hoveroff', (evt) => this.setHover(evt.object, false));
    this.selector.addEventListener('select', (evt) => this.setSelected(evt.object, true));
    this.selector.addEventListener('deselect', (evt) => this.setSelected(evt.object, false));

    this.defaultTarget = this.selector.selectable[0];
    this.target = this.defaultTarget;

    this.setDirty();

    window.onload = (evt) => this.setSize();
    window.addEventListener('resize', (evt) => this.setSize(), false);
    requestAnimationFrame(() => this.setSize());

    const resizeObserver = new ResizeObserver(() => this.resizeCanvasToDisplaySize());
    resizeObserver.observe(this.domElement, {box: 'content-box'});

    this.animate();
  }

  setDirty() {
    this.needsRender = true;
  }

  setSize(w, h) {
    if (w === undefined) w = this.domElement.offsetWidth;
    if (h === undefined) h = this.domElement.clientHeight;
    if (this.camera.type == 'OrthographicCamera') {
      this.camera.right = this.camera.left + w * (this.camera.top - this.camera.bottom) / h;
    } else {
      this.camera.aspect = w / h;
    }
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.setDirty();
  }

  resizeCanvasToDisplaySize() {
    const width = this.domElement.clientWidth;
    const height = this.domElement.clientHeight;
    this.setSize(width, height);
  }

  render() {
    toggleDebugMode(this.scene, this.debugMode);
    this.renderer.render(this.scene, this.camera);
    this.needsRender = false;
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.animator.update();

    const targetPos = new THREE.Vector3();
    this.target.getWorldPosition(targetPos);

    if (this.camera.follow) {
      this.controls.target.lerp(targetPos, 0.1);
      if (this.camera.position.distanceTo(this.controls.target) > this.camera.followDistance) {
        const followBehind = this.controls.target.clone()
          .sub(this.camera.position)
          .normalize()
          .multiplyScalar(this.camera.followDistance)
          .sub(this.controls.target)
          .negate();
        this.camera.position.lerp(followBehind, 0.5);
        this.setDirty();
      }
    }

    this.dirLight.position.set(targetPos.x + 3, targetPos.y + 10, targetPos.z + 10);
    this.dirLight.target = this.target;

    if (this.controls.update()) this.setDirty();

    if (this.camera.freezeAngle) {
      const off = new THREE.Vector3();
      off.add(this.controls.target).sub(this.controlTargetPos);
      off.setComponent(1, 0);
      if (off.lengthSq() > 0) {
        this.camera.position.add(off);
        this.setDirty();
      }
    }
    this.controlTargetPos.copy(this.controls.target);

    if (this.needsRender) this.render();
  }

  saveImage() {
    this.render();
    const imageData = this.renderer.domElement.toDataURL();
    downloadDataUri('brax.png', imageData);
  }

  saveScene() {
    downloadFile('system.json', JSON.stringify(this.system));
  }

  setDebugMode(val) {
    this.debugMode = val;
  }

  setHover(object, hovering) {
    this.setDirty();
    if (!object.selected) {
      object.traverse(function(child) {
        if (child instanceof THREE.Mesh) {
          child.material = hovering ? hoverMaterial : child.baseMaterial;
        }
      });
    }
    if (object.name in this.bodyFolders) {
      const titleElement = this.bodyFolders[object.name].domElement.querySelector('.title');
      if (titleElement) {
        titleElement.style.backgroundColor = hovering ? '#2fa1d6' : '#000';
      }
    }
  }

  setSelected(object, selected) {
    object.selected = selected;
    this.target = selected ? object : this.defaultTarget;
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = selected ? selectMaterial : child.baseMaterial;
      }
    });
    if (object.name in this.bodyFolders) {
      if (object.selected) this.bodyFolders[object.name].open();
      else this.bodyFolders[object.name].close();
    }
    this.setDirty();
  }

  loadNewTrajectory(newSystem) {
    const newTrajectory = createTrajectory(newSystem);
    this.animator.load(newTrajectory, {});
    this.setDirty();
  }
}

// --- Live Frame Streaming Support ---

let viewerInstance = null;

function handleFrame(frameJson) {
  let parsed;
  try {
    // The console log is in the user prompt, so we can be sure this is being called.
    console.log(frameJson); 
    parsed = JSON.parse(frameJson);
  } catch (e) {
    console.error("JSON parse error:", e);
    return;
  }

  if (!parsed.x || !parsed.x.pos || !parsed.x.rot) {
    console.warn("Invalid frame: missing x/pos/rot", parsed);
    return;
  }

  const frame = parsed.x;
  const scene = viewerInstance.scene;
  const env_id = parsed.env_id;

  if (!frame.pos || !frame.rot) {
    console.warn('Invalid frame format: missing pos or rot');
    return;
  }

  // Get the number of links per environment. If not calculated (e.g. single env),
  // it will be 0, and we'll use the global index.
  const numLinks = viewerInstance.numLinksPerEnv || 0;

  Object.entries(viewerInstance.system.geoms).forEach(function ([name, geoms]) {
    // For batched envs, only process geoms belonging to the current frame's env_id.
    if (numLinks > 0) {
        let name_matches_env = false;
        if (env_id === 0) {
            // Env 0 geoms should NOT have a numeric suffix.
            if (!/_\d+$/.test(name)) {
                name_matches_env = true;
            }
        } else {
            // Other env geoms should end with the corresponding `_env_id`.
            if (name.endsWith(`_${env_id}`)) {
                name_matches_env = true;
            }
        }
        if (!name_matches_env) {
            return; // Skip geoms from other environments.
        }
    }

    const global_link_idx = geoms[0].link_idx;
    if (global_link_idx == null || global_link_idx < 0) return;

    let local_link_idx = global_link_idx;
    if (numLinks > 0) {
      // Map the global link index from the concatenated system to the local
      // index of the single-environment state data.
      local_link_idx = global_link_idx % numLinks;
    }

    const groupName = name.replaceAll('/', '_');
    const group = scene.getObjectByName(groupName);
    if (!group) {
      // It's possible not all geoms are being updated, so this is not a fatal error.
      return;
    }

    if (local_link_idx >= frame.pos.length) {
      console.warn(`Index out of bounds: local_link_idx ${local_link_idx} for frame.pos of length ${frame.pos.length}`);
      return;
    }

    const pos = frame.pos[local_link_idx];
    group.position.fromArray(pos);

    const rot = frame.rot[local_link_idx];
    group.quaternion.set(rot[1], rot[2], rot[3], rot[0]);  // [w, x, y, z] → [x, y, z, w]
  });

  viewerInstance.setDirty();
}


// Set up WebSocket for live frames (no buffering)
function setupLiveFrameWebSocket(system, viewer) {
  viewerInstance = viewer;
  const ws = new WebSocket(`ws://${window.location.host}/ws/frame`);

  // Add status light for live frame WS
  const frameStatus = { status: 'Connecting' };
  const frameFolder = viewer.gui.addFolder('Live Frame WS Status');
  const frameCtrl = frameFolder.add(frameStatus, 'status').listen();
  frameFolder.domElement.style.color = 'red';
  frameFolder.close();

  function setFrameStatus(text, color) {
    frameStatus.status = text;
    frameCtrl.updateDisplay();
    frameFolder.domElement.style.color = color;
  }

  ws.onopen = () => {
    console.log('Connected to live frame WebSocket:', window.location.host);
    setFrameStatus('Connected', 'green');
  };
  ws.onmessage = (event) => {
    handleFrame(event.data);
  };
  ws.onclose = () => {
    console.log('WebSocket closed');
    setFrameStatus('Disconnected', 'yellow');
  };
  ws.onerror = (e) => {
    console.error('WebSocket error:', e);
    setFrameStatus('Error', 'red');
  };
}

function setupControlWebSocket(viewer) {
  const ws = new WebSocket(`ws://${window.location.host}/ws/control`);
  window.control_ws = ws; // Make ws globally accessible for the GUI callback

  // Add status light for control WS
  const controlStatus = { status: 'Connecting' };
  const controlFolder = viewer.gui.addFolder('Control WS Status');
  const controlCtrl = controlFolder.add(controlStatus, 'status').listen();
  controlFolder.domElement.style.color = 'red';
  controlFolder.close();

  function setControlStatus(text, color) {
    controlStatus.status = text;
    controlCtrl.updateDisplay();
    controlFolder.domElement.style.color = color;
  }

  ws.onopen = () => {
    console.log('[Control] Connected');
    setControlStatus('Connected', 'green');
  };
  ws.onclose = () => {
    console.warn('[Control] Disconnected');
    setControlStatus('Disconnected', 'yellow');
  };
  ws.onerror = (e) => {
    console.error('[Control] Error:', e);
    setControlStatus('Error', 'red');
  };

  ws.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch (err) {
      console.error('[Control] Invalid JSON:', err);
      return;
    }

    if (msg.type === 'refresh_web') {
      window.location.reload();
    }
  };
}

export {Viewer, setupLiveFrameWebSocket, setupControlWebSocket};