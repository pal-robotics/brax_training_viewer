# Config

Configuration class for the viewer system.

## Purpose

Config centralizes all configuration parameters for the BraxViewer system, including network settings, logging levels, environment parameters, and layout configuration. It provides a single point of configuration management.

## Constructor

```python
Config(
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "warning",
    server_log_level: str = "warning",
    xml: Optional[str] = None,
    unbatched_stream: bool = True,
    num_envs: Optional[int] = None,
    grid_dims: Optional[tuple] = None,
    env_offset: Optional[tuple] = None
)
```

## Parameters

- `host` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Server host address (default: "127.0.0.1")
- `port` ([int](https://docs.python.org/3/c-api/long.html)): Server port (default: 8000)
- `log_level` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Logging level for the viewer (default: "warning")
- `server_log_level` ([str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)): Logging level for the server (default: "warning")
- `xml` (Optional[[str](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)]): XML model string (default: None)
- `unbatched_stream` ([bool](https://docs.python.org/3/c-api/bool.html)): Whether to use unbatched streaming (default: True)
- `num_envs` (Optional[[int](https://docs.python.org/3/c-api/long.html)]): Number of parallel environments (default: None)
- `grid_dims` (Optional[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)]): Grid dimensions for environment layout (default: None)
- `env_offset` (Optional[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)]): Environment offset for layout (default: None)

## Properties

All constructor parameters are accessible as properties with the same names and types. 