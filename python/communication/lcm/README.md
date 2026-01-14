# LCM Communication Guide

This directory provides a typed LCM interface for controlling the ARX5 Cartesian controller.

## Quick Start

1) Start the LCM server (robot side):

```bash
python python/communication/lcm/lcm_server.py <MODEL> <INTERFACE> --address 239.255.76.67 --port 7667 --ttl 1
```

Example:
```bash
python python/communication/lcm/lcm_server.py X5 can0
```

2) Use the client (control side):

```python
import numpy as np
from communication.lcm.lcm_client import Arx5LcmClient

client = Arx5LcmClient(address="239.255.76.67", port=7667, ttl=1)
state = client.get_state()

target_pose = state["ee_pose"].copy()
target_pose[0] += 0.01  # small +X move
client.set_ee_pose(target_pose, gripper_pos=None, preview_time=0.1)
```

## API Overview

Client class: `Arx5LcmClient` in `python/communication/lcm/lcm_client.py`.

Supported commands:
- `get_state()`: returns a dict with timestamp, ee_pose, joint states, and gripper states.
- `set_ee_pose(pose_6d, gripper_pos=None, preview_time=None)`: sends a cartesian command.
- `reset_to_home()`: resets robot to home.
- `set_to_damping()`: sets damping mode.
- `get_gain()` / `set_gain(gain_dict)`: gains round trip.

### Optional Arguments Behavior

- `preview_time=None`: server uses the controller default preview time (C++ side default).
- `gripper_pos=None`: client sends `NaN` and server keeps the current gripper position.

## LCM Message Schema

The LCM types live in `python/communication/lcm/defs/`.
If you change `.lcm` files, regenerate the Python types:

```bash
lcm-gen -p --ppath python/communication/lcm python/communication/lcm/defs/arx5_command_t.lcm
```

Keep client and server schemas in sync, or decoding will fail.

## Safety Notes (Real Robot)

- Start with small cartesian deltas (e.g., 0.01 m) and a short preview time (e.g., 0.1 s).
- Verify your gripper calibration and joint limits before testing motion commands.
- Use `reset_to_home()` and `set_to_damping()` if commands time out or you need to stop.
