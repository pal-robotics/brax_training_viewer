# Brax Rendering Integration

This document describes the minimal changes made to Brax to support conditional rendering during PPO training, addressing the performance issues discussed in the JAX conditional callback analysis.

## Overview

The integration adds two simple parameters to the PPO training function:
- `should_render`: A boolean flag to control rendering
- `render_fn`: A callback function that receives environment states

This approach minimizes changes to Brax source code while providing the functionality needed for efficient conditional rendering.

## Changes Made

### 1. PPO Training Function (`braxviewer/brax/brax/training/agents/ppo/train.py`)

Added two new parameters to the `train()` function:

```python
def train(
    # ... existing parameters ...
    # rendering
    should_render: bool = False,
    render_fn: Optional[Callable[[envs.State], None]] = None,
    # ... rest of parameters ...
):
```

### 2. Acting Functions (`braxviewer/brax/brax/training/acting.py`)

Modified `actor_step()` and `generate_unroll()` to support rendering:

```python
def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
    should_render: bool = False,
    render_fn: Optional[Callable[[State], None]] = None,
) -> Tuple[State, Transition]:
    # ... existing logic ...
    
    # Conditional rendering
    if should_render and render_fn is not None:
        render_fn(nstate)
    
    # ... rest of function ...
```

## Usage

### Basic Example

```python
from braxviewer.WebViewer import WebViewer
from braxviewer.brax.brax.training.agents.ppo import train as ppo

# Create environment and viewer
env = YourEnvironment()
viewer = WebViewer(xml=your_xml_model)
viewer.run()

# Create render function
def render_fn(state):
    viewer.send_frame(state)

# Train with rendering
make_policy_fn, params, _ = ppo.train(
    environment=env,
    eval_env=env,
    # ... other training parameters ...
    should_render=True,
    render_fn=render_fn,
)
```

### Conditional Rendering

You can control rendering based on training progress:

```python
def create_conditional_render_fn(viewer, render_frequency=100):
    step_count = 0
    
    def render_fn(state):
        nonlocal step_count
        step_count += 1
        if step_count % render_frequency == 0:
            viewer.send_frame(state)
    
    return render_fn

# Use conditional rendering
render_fn = create_conditional_render_fn(viewer, render_frequency=50)
```

## Performance Benefits

This implementation addresses the JAX conditional callback performance issues by:

1. **Minimal Brax Changes**: Only adds two parameters to existing functions
2. **No Complex Wrappers**: Avoids the need for complex environment wrappers
3. **Direct Integration**: Rendering is integrated directly into the training loop
4. **Conditional Execution**: Rendering only occurs when explicitly enabled

## Comparison with Previous Approach

### Before (ViewerWrapper)
- Required wrapping the environment
- Always executed rendering logic
- More complex integration
- Performance issues in vmap contexts

### After (Direct Integration)
- No environment wrapping needed
- Conditional rendering with simple flags
- Minimal code changes
- Better performance characteristics

## Examples

See the following example files:
- `demo/simple_rendering_example.py`: Basic rendering demonstration
- `demo/cartpole_batched.py`: Batched environment training with rendering
- `demo/conditional_rendering_example.py`: Advanced conditional rendering patterns

## Technical Details

The implementation follows the "Pass-the-Flag" pattern discussed in the JAX analysis:

1. **Scalar Control**: `should_render` is a scalar boolean passed to the training function
2. **Direct Integration**: Rendering logic is integrated into `actor_step()` 
3. **Conditional Execution**: Rendering only occurs when both `should_render=True` and `render_fn` is provided
4. **No vmap Issues**: The rendering logic is not subject to vmap's cond→select conversion

This approach provides a clean, efficient solution for conditional rendering in Brax PPO training while maintaining the performance benefits of JAX's vectorized execution. 