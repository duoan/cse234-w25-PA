# PyTorch Development Container

This directory contains configuration for a PyTorch development container using VS Code's Dev Containers feature.

## Quick Start

### Using VS Code Dev Containers (Recommended)
1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the workspace folder in VS Code
3. Click the green button in the bottom-left corner (><) and select "Reopen in Container"
4. VS Code will build and start the container automatically

### Manual Docker Compose
```bash
cd .devcontainer
docker-compose up -d
docker-compose exec pytorch-dev bash
```

## What's Included

- **PyTorch** with CUDA 12.1 support
- **Python 3.11** via Miniconda
- **Jupyter Lab** for notebook development
- **VS Code Extensions**: Python, Pylance, Jupyter, Make Tools
- **Development Tools**: git, build-essential, curl, wget, vim
- **Common Libraries**: numpy, scipy, pandas, scikit-learn, matplotlib

## Environment Details

- **Base Image**: Official Microsoft Miniconda dev container
- **GPU Support**: CUDA 12.1 (if available on host)
- **Shared Memory**: 2GB (for PyTorch DataLoaders)
- **Python Version**: 3.11

## Usage Notes

- The container mounts your workspace at `/workspaces`
- SSH keys are automatically available in the container
- Changes made inside the container persist on your host machine
- To rebuild after updating dependencies, use the command palette: "Dev Containers: Rebuild Container"

## Customization

Edit `devcontainer.json` to:
- Add more VS Code extensions
- Adjust conda/pip packages
- Modify mount points or environment variables
- Change the Python version (update the base image tag)

For more information, see [VS Code Dev Containers documentation](https://code.visualstudio.com/docs/devcontainers/containers).
