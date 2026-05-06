# How to Run the Robot (Obstacle Avoidance)

The pipeline is:

```
Camera → vision_node → [obstacle_node] → nav_node → robot_driver_node → Arduino → Motors
```

> `obstacle_node` does not exist yet. You need to create it or wire your own vision code into `/semantic/decision`.  
> See the **Architecture** section at the bottom for what each node publishes.

---

## Prerequisites

- Orin Nano has ROS 2 installed and sourced
- Arduino is flashed and connected via USB (default `/dev/ttyUSB0`)
- Camera is connected (default device index `0`)
- Workspace is built (see Step 1)

---

## Step 1 — Build the workspace

```bash
cd ~/DiskD/RoboticsWorks/scs_robot
colcon build --packages-select semantic_comm_runtime
source install/setup.bash
```

---

## Step 2 — Verify Arduino serial connection

Plug in the Arduino and check which port it appears on:

```bash
ls /dev/ttyUSB*
# or
ls /dev/ttyACM*
```

Test that you can write to it (replace the port if different):

```bash
echo "M127,127,127" > /dev/ttyUSB0
```

That is the "stop all motors" command. Wheels should not spin.

---

## Step 3 — Test the motor driver alone (no camera)

Open **Terminal 1**:

```bash
source install/setup.bash
ros2 run semantic_comm_runtime robot_driver_node \
  --ros-args -p robot_serial_port:=/dev/ttyUSB0
```

Open **Terminal 2** — send a forward command:

```bash
source install/setup.bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

The robot should move forward. To stop:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

**Do not proceed until wheels spin correctly.**

---

## Step 4 — Test the camera / vision node

Open **Terminal 1**:

```bash
source install/setup.bash
ros2 run semantic_comm_runtime vision_node \
  --ros-args -p camera_id:=0
```

Open **Terminal 2** — watch what it detects:

```bash
source install/setup.bash
ros2 topic echo /vision/centroids
```

Put a coloured object in front of the camera. You should see `[cx, cy]` values printed.  
If nothing appears, tune the HSV parameters (see `vision_node.py`).

---

## Step 5 — Test the nav node manually

The nav node reads `/semantic/decision` (Int32) and publishes `/cmd_vel`.

Decision codes:
| Value | Behaviour |
|-------|-----------|
| `0`   | Move forward |
| `1`   | Turn left |
| `2`   | Turn right |
| `3`   | Stop |

Run nav_node:

```bash
source install/setup.bash
ros2 run semantic_comm_runtime nav_node
```

Send a manual decision in another terminal:

```bash
# Go forward
ros2 topic pub /semantic/decision std_msgs/msg/Int32 "{data: 0}"

# Stop
ros2 topic pub /semantic/decision std_msgs/msg/Int32 "{data: 3}"
```

---

## Step 6 — Wire your vision code to the decision topic

Your external vision code that detects obstacles needs to publish to `/semantic/decision` (std_msgs/Int32).

The simplest approach — add one publisher to your existing vision script:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# Inside your detection loop, after you decide what to do:
# decision = 0  # forward — no obstacle
# decision = 3  # stop    — obstacle ahead
# decision = 1  # turn left
# decision = 2  # turn right

# Publish it:
self.pub.publish(Int32(data=decision))
```

---

## Step 7 — Run everything together

Once each piece works individually, run all nodes at the same time.

**Terminal 1 — robot driver:**
```bash
source install/setup.bash
ros2 run semantic_comm_runtime robot_driver_node \
  --ros-args -p robot_serial_port:=/dev/ttyUSB0
```

**Terminal 2 — nav node:**
```bash
source install/setup.bash
ros2 run semantic_comm_runtime nav_node
```

**Terminal 3 — vision node:**
```bash
source install/setup.bash
ros2 run semantic_comm_runtime vision_node \
  --ros-args -p camera_id:=0
```

**Terminal 4 — your obstacle detection code:**  
Run your external vision script here. Make sure it publishes to `/semantic/decision`.

---

## Monitoring / Debugging

Check all active topics:
```bash
ros2 topic list
```

Watch the decision stream:
```bash
ros2 topic echo /semantic/decision
```

Watch cmd_vel:
```bash
ros2 topic echo /cmd_vel
```

Check node graph:
```bash
ros2 node list
ros2 run rqt_graph rqt_graph
```

---

## Architecture Reference

```
/vision/centroids  (Float32MultiArray)
    ← published by: vision_node
    → consumed by: your obstacle detection code

/semantic/decision  (Int32: 0=fwd, 1=left, 2=right, 3=stop)
    ← published by: your obstacle detection code
    → consumed by: nav_node

/cmd_vel  (geometry_msgs/Twist)
    ← published by: nav_node
    → consumed by: robot_driver_node

Serial "M<w0>,<w1>,<w2>\n"  (0–255 per wheel, 127=stop)
    ← sent by: robot_driver_node
    → received by: Arduino → Motors
```
