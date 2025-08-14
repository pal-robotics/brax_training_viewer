# Troubleshooting

This guide covers common installation issues and their solutions.

## Common Issues

### Port Conflicts

**Problem**: `ERROR: [Errno 48] error while attempting to bind on address ('x.x.x.x', xxxx): [errno 48] address already in use`

**Solution**: Check availability of a port. A port may still be occupied due to unexpected quit of a thread that uses the port.

## Getting Help

If you continue to experience issues:

1. Check the [Installation Guide](../installation) for the complete setup process
2. [Open an issue](https://github.com/pal-robotics/brax_training_viewer/issues) on GitHub 