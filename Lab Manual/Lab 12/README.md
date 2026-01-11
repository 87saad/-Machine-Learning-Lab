# Object Detection Project

A Unity-based object detection application with a Python backend server.

## Project Structure

- **APP/** - Unity application for object detection visualization
- **SERVER/** - Python server using YOLOv8 for object detection

## Requirements

### Unity App
- Unity Editor (version specified in ProjectSettings)
- Visual Studio or compatible IDE

### Python Server
- Python 3.x
- Dependencies managed via uv (see `SERVER/pyproject.toml`)

## Setup

### Server Setup
```bash
cd SERVER
# Install dependencies using uv
uv sync
```

### Running the Server
```bash
cd SERVER
python server.py
```

### Unity App
1. Open the `APP` folder in Unity Editor
2. Build and run the project
