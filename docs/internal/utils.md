# Utils

Utility functions for state conversion and processing.

## Purpose

Utils provide functions for converting Brax states to JSON-serializable formats and handling batched state processing. These functions are essential for state transmission between the training process and the web viewer.

## Functions

### state_to_dict(state: State, index: int = 0, unbatched: bool = True) -> dict

Convert Brax State to JSON-serializable dictionary.

**Parameters:**
- `state` ([State](https://github.com/google/brax/blob/main/brax/envs/base.py)): Brax environment state
- `index` ([int](https://docs.python.org/3/c-api/long.html)): Index for batched states (default: 0)
- `unbatched` ([bool](https://docs.python.org/3/c-api/bool.html)): Whether to process as unbatched (default: True)

**Returns:**
- `dict`: JSON-serializable dictionary representation of the state

### unbatch_state(state: State, index: int) -> State

Extract index-th sample from batched state.

**Parameters:**
- `state` ([State](https://github.com/google/brax/blob/main/brax/envs/base.py)): Batched Brax state
- `index` ([int](https://docs.python.org/3/c-api/long.html)): Index to extract

**Returns:**
- [`State`](https://github.com/google/brax/blob/main/brax/envs/base.py): Single environment state

### _physics_state_to_dict(phys_state: BaseState) -> dict

Convert physics state to dictionary.

**Parameters:**
- `phys_state` ([BaseState](https://github.com/google/brax/blob/main/brax/base.py)): Physics state object

**Returns:**
- `dict`: Dictionary representation of physics state

**Note:** This is an internal helper function. 