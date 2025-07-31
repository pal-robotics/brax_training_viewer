# Internal Implementation

This section contains documentation for the internal implementation of braxviewer. These components are not intended for direct use by end users, but are documented here for developers who need to understand or extend the library.

## Overview

The internal implementation is organized into several key components:

- **Configuration Management**: Core configuration classes and settings
- **Network Layer**: WebSocket streaming, server management, and connection handling
- **Data Processing**: State conversion, serialization, and utility functions
- **Rendering Pipeline**: HTML generation and visualization utilities

## Components

### Core Infrastructure
- [Config](internal/config.md) - Configuration management and settings
- [Server](internal/server.md) - FastAPI web server implementation
- [ConnectionManager](internal/connectionmanager.md) - WebSocket connection management

### Data Processing
- [StateStreamer](internal/statestreamer.md) - Real-time data streaming
- [Sender](internal/sender.md) - Base data sending functionality
- [Utils](internal/utils.md) - State conversion and utility functions

### Rendering
- [HTML Utils](internal/html-utils.md) - HTML generation and template rendering

## Developer Notes

These components are designed to work together to provide the high-level user interface. If you're extending braxviewer, understanding these components will help you integrate new features effectively.

For most use cases, the [Python API](python-api.md) provides all the functionality you need. 