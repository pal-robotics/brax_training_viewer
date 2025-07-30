# Contributing

This page is for contributors who plan to develop or extend BraxViewer.

## Developer setup (with Brax as a submodule)

For development, you can work against the Brax submodule to modify the upstream code that BraxViewer integrates with.

```bash
# Download modified Brax as submodule
git submodule update --init --recursive

# Install modified Brax in editable mode
pip install -e ./brax

# Install BraxViewer in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

:::{note}
- Use editable installs (`-e`) so changes are picked up without reinstallation.
:::

## Code layout and where to make changes

- `braxviewer/brax/**`: a minimal mirror of selected Brax training components (e.g., PPO) adapted to import paths under `braxviewer`. This copy exists to provide modifed Brax package.
- `brax/**` (submodule): a fork of original Brax codebase. If you need to change core Brax functionality, make changes here. And then, copy the file to `braxviewer/brax/**` to deliver with braxviewer.

## Recommended reading

- Internal source code overview: see `Internal Implementation`
