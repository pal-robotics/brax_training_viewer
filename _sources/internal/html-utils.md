# HTML Utils

Utilities for HTML rendering and visualization.

## Purpose

HTML utils provide functions for generating HTML content for the BraxViewer web interface. These functions handle system serialization, template rendering, and HTML file generation.

## Functions

### render_from_json(sys: str, height: Union[int, str], colab: bool, base_url: Optional[str]) -> str

Render HTML from JSON system string.

**Parameters:**
- `sys` (str): JSON string representation of the system
- `height` (Union[int, str]): Height of the render window
- `colab` (bool): Whether to use CSS styles for colab
- `base_url` (Optional[str]): Base URL for serving visualizer files

**Returns:**
- `str`: HTML string for the visualizer

### render(sys: System, states: List[State], height: Union[int, str] = 480, colab: bool = True, base_url: Optional[str] = None) -> str

Render HTML from system and states.

**Parameters:**
- `sys` (System): Brax system object
- `states` (List[State]): List of system states to render
- `height` (Union[int, str]): Height of the render window (default: 480)
- `colab` (bool): Whether to use CSS styles for colab (default: True)
- `base_url` (Optional[str]): Base URL for serving visualizer files (default: None)

**Returns:**
- `str`: HTML string for the visualizer

### save(path: str, sys: System, states: List[State])

Save trajectory as HTML file.

**Parameters:**
- `path` (str): File path to save the HTML
- `sys` (System): Brax system object
- `states` (List[State]): List of system states to save 