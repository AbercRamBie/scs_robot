# ROS 2 Migration Guide

## Purpose

This document explains how to make the whole repository operate as a clean ROS 2 project instead of a mix of:

- ROS 2 runtime code in `src/simulation`
- plain Python and Torch code in `src/ml/ml`
- hardcoded local paths between the two

The goal is not to rewrite the ML logic into ROS APIs. The correct target is:

- all executable runtime pieces live in ROS 2 packages
- ML code is packaged and installed inside the ROS 2 workspace
- ROS 2 nodes import ML code through normal package imports
- launch files and nodes resolve models and assets through package paths
- no `sys.path.insert(...)`
- no absolute machine-specific paths

## Current State Summary

The repository already contains a ROS 2 workspace layout:

- `src/simulation` is an `ament_python` package with ROS 2 nodes and a launch file.
- `src/ml` is also an `ament_python` package, but it currently behaves more like a plain Python library bucket.
- simulation nodes still depend on hardcoded paths and manual import hacks to reach ML code and checkpoints.

So this is not a fresh migration from non-ROS code. It is a cleanup and completion migration.

## Target End State

After migration, the workspace should look like this:

```text
scs_robot/
  src/
    semcomm_core/
      package.xml
      setup.py
      setup.cfg
      resource/
      semcomm_core/
        __init__.py
        awgn.py
        channel_layer.py
        config.py
        decoder.py
        encoder.py
        occupancy.py
        rayleigh.py
        results.py
        run_train.py
        separation_baseline.py
        snr_sweep.py
        sweep.py
        trainer.py
        vib.py
    semantic_comm_runtime/
      package.xml
      setup.py
      setup.cfg
      resource/
      launch/
      config/
      assets/
        robot/
        world/
      semantic_comm_runtime/
        __init__.py
        encoder_node.py
        channel_node.py
        decoder_node.py
        nav_node.py
  artifacts/
    checkpoints/
    results/
```

You can keep the existing package names if you want, but using clear names such as `semcomm_core` and `semantic_comm_runtime` avoids confusion between the ROS package name and the Python subpackage name.

## Recommended Strategy

Use two ROS 2 Python packages inside the same workspace.

1. `semcomm_core`
   Contains all reusable ML, channel, training, evaluation, and utility code.

2. `semantic_comm_runtime`
   Contains ROS 2 nodes, launch files, simulator assets, runtime parameters, and any ROS message handling.

This is the right design because:

- training code stays reusable and testable
- ROS nodes stay thin and focused on subscriptions, publications, and runtime orchestration
- imports become stable and portable
- packaging works with `colcon build`

## Migration Plan Overview

Perform the migration in this order:

1. Create or rename the ML package into a clean ROS 2 Python package.
2. Move ML source files under a dedicated import package.
3. Fix imports inside the ML package.
4. Make the runtime package depend on the ML package through normal imports.
5. Remove all `sys.path.insert(...)` usage.
6. Replace absolute paths with ROS 2 package-share path resolution.
7. Register optional training and evaluation entry points.
8. Build and test the workspace with `colcon`.
9. Run launch and node-level validation.

## Step 1: Create a Branch

Run the migration in a separate branch.

```bash
cd ~/DiskD/RoboticsWorks/scs_robot
git checkout -b ros2-full-migration
```

## Step 2: Decide Package Names

You have two options.

### Option A: Minimal rename

- keep `ml` as the ML ROS 2 package name
- keep `simulation` as the runtime ROS 2 package name

This is lower effort but the names are generic.

### Option B: Clear package names

- rename `ml` to `semcomm_core`
- rename `simulation` to `semantic_comm_runtime`

This is the better long-term structure and is what this guide assumes.

If you prefer minimal change, apply the same steps but keep the old names.

## Step 3: Restructure the ML Package

Current ML code sits here:

- `src/ml/ml/*.py`

Create a dedicated ROS 2 Python package folder:

```text
src/semcomm_core/
  package.xml
  setup.py
  setup.cfg
  resource/
  semcomm_core/
```

Move the Python files from `src/ml/ml/` into `src/semcomm_core/semcomm_core/`.

Files to move:

- `awgn.py`
- `channel_layer.py`
- `config.py`
- `decoder.py`
- `encoder.py`
- `occupancy.py`
- `rayleigh.py`
- `results.py`
- `run_train.py`
- `separation_baseline.py`
- `snr_sweep.py`
- `sweep.py`
- `trainer.py`
- `vib.py`

Keep `__init__.py` in the package root.

## Step 4: Update ML Package Metadata

Edit `package.xml` for the ML package.

Minimum requirements:

- set a real description
- declare runtime Python dependencies if you want ROS package metadata to be meaningful
- keep `ament_python` as the build type

Example shape:

```xml
<package format="3">
  <name>semcomm_core</name>
  <version>0.1.0</version>
  <description>Semantic communication core models, training, and evaluation utilities.</description>
  <maintainer email="subashram773@gmail.com">subash</maintainer>
  <license>Apache-2.0</license>

  <exec_depend>python3-numpy</exec_depend>
  <exec_depend>python3-torch</exec_depend>

  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Notes:

- `torch` is often installed through pip or conda, not apt, so your environment setup must still handle that.
- package.xml metadata is still useful even if some Python packages are installed outside apt.

## Step 5: Update ML setup.py

Use `find_packages()` to install the Python package and expose command-line scripts for training and evaluation.

Recommended console scripts:

- `train_semcomm`
- `eval_baseline`
- `eval_snr_sweep`

Example:

```python
from setuptools import find_packages, setup

package_name = 'semcomm_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='subash',
    maintainer_email='subashram773@gmail.com',
    description='Semantic communication core models, training, and evaluation utilities.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'train_semcomm = semcomm_core.run_train:main',
            'eval_baseline = semcomm_core.separation_baseline:main',
            'eval_snr_sweep = semcomm_core.snr_sweep:main',
        ],
    },
)
```

If a file does not currently define `main()`, add one. Do not expose scripts that execute training on import.

## Step 6: Fix ML Internal Imports

After moving the files, fix imports so they always use package-qualified imports.

Example patterns:

Replace:

```python
from encoder import SemanticEncoder
from vib import vib_loss
```

With:

```python
from semcomm_core.encoder import SemanticEncoder
from semcomm_core.vib import vib_loss
```

This is the step that makes the ML package installable and importable from ROS 2 nodes without path hacks.

## Step 7: Restructure the Runtime Package

Current runtime code sits here:

- `src/simulation/simulation/*.py`
- `src/simulation/launch/*.launch.py`
- `src/simulation/assets/...`

Create or rename to:

```text
src/semantic_comm_runtime/
  package.xml
  setup.py
  setup.cfg
  resource/
  launch/
  config/
  assets/
    robot/
    world/
  semantic_comm_runtime/
```

Move the runtime Python files into the new Python package folder.

## Step 8: Update Runtime Package Dependencies

The runtime package should explicitly depend on:

- `rclpy`
- `sensor_msgs`
- `std_msgs`
- `launch`
- `launch_ros`
- the ML package, either `ml` or `semcomm_core`

Your runtime `package.xml` should include the ROS dependencies it actually uses.

Example shape:

```xml
<exec_depend>rclpy</exec_depend>
<exec_depend>sensor_msgs</exec_depend>
<exec_depend>std_msgs</exec_depend>
<exec_depend>launch</exec_depend>
<exec_depend>launch_ros</exec_depend>
<exec_depend>semcomm_core</exec_depend>
```

If you later add custom messages, then add `rosidl_default_generators` and a message package. That is not required for the current migration.

## Step 9: Remove sys.path Hacks From Nodes

This is one of the most important fixes.

Current runtime nodes manually inject paths, which makes the package machine-specific and fragile.

Replace code like this:

```python
import sys
sys.path.insert(0, '/home/subash/miniconda3/envs/semcomm/lib/python3.11/site-packages')
sys.path.insert(0, '/home/subash/DiskD/RoboticsWorks/scs_robot/src/ml/ml')
```

With normal imports:

```python
from semcomm_core.encoder import SemanticEncoder
from semcomm_core.vib import reparametrize
```

If a node imports `packages.models.encoder`, fix that to the actual installed package path.

## Step 10: Resolve Checkpoints and Assets Through ROS 2 Package Paths

Do not use absolute paths such as:

```python
'/home/subash/DiskD/RoboticsWorks/scs_robot/artifacts/checkpoints/encoder_snr10.pth'
```

Instead, use one of these strategies.

### Strategy A: Use a node parameter

This is the best option for model files.

Declare a parameter:

```python
self.declare_parameter('encoder_checkpoint', '')
checkpoint_path = self.get_parameter('encoder_checkpoint').get_parameter_value().string_value
```

Pass the path from the launch file.

This keeps large model artifacts outside the installed package while still making runtime configuration clean.

### Strategy B: Put a small default model into package share

Only do this if you truly want a baked-in default model.

Install it through `data_files` in `setup.py`, then resolve with:

```python
from ament_index_python.packages import get_package_share_directory

share_dir = get_package_share_directory('semantic_comm_runtime')
```

### Strategy C: Use package share for SDF assets

For robot and world files, package share is the correct choice.

In launch files, replace hardcoded paths with:

```python
from ament_index_python.packages import get_package_share_directory
import os

share_dir = get_package_share_directory('semantic_comm_runtime')
robot_sdf = os.path.join(share_dir, 'assets', 'robot', 'semantic_robot.sdf')
world = os.path.join(share_dir, 'assets', 'world', 'semantic_world.sdf')
```

## Step 11: Clean Up setup.py for Runtime Assets

The runtime package already installs launch files and SDF files through `data_files`. Keep that pattern, but ensure the package name and folder names match after the rename.

Make sure these are installed:

- launch files
- runtime YAML config files
- robot SDF assets
- world SDF assets

If you add a config folder, include it in `data_files` as well.

## Step 12: Add Runtime Parameters

Move runtime configuration out of source code and into parameters.

At minimum, parameterize:

- checkpoint path
- SNR
- bottleneck dimension
- grid size
- topic names if needed

Create a YAML file such as:

```text
src/semantic_comm_runtime/config/runtime.yaml
```

Example shape:

```yaml
encoder_node:
  ros__parameters:
    encoder_checkpoint: /absolute/or/relative/path/to/encoder_snr10.pth
    grid_size: 64
    bottleneck_dim: 2

channel_node:
  ros__parameters:
    snr_db: 5.0
```

Then load it from the launch file.

## Step 13: Update Launch Files

The launch file should:

- resolve package share directories with `get_package_share_directory`
- pass parameter files to nodes
- avoid all hardcoded home-directory paths
- keep Gazebo startup separate from model path assumptions

Typical launch responsibilities:

1. locate world asset
2. locate robot asset
3. locate runtime parameter YAML
4. start Gazebo
5. spawn the robot
6. start encoder, channel, decoder, and navigation nodes

## Step 14: Make Training Runnable Inside the ROS 2 Workspace

If you want the whole project to be in ROS 2, training should also be invokable after sourcing the workspace.

That does not mean training must become a ROS node.

It means training should be installable and executable as a ROS 2 package console script.

Target usage:

```bash
source install/setup.bash
ros2 run semcomm_core train_semcomm
```

If you want to pass configs, prefer command-line arguments or parameter files read by plain Python.

## Step 15: Install Python Dependencies Correctly

ROS 2 packaging alone does not solve heavy Python ML dependencies.

For this repository, define the environment clearly.

Recommended approach:

1. Create a Python environment compatible with your ROS 2 distribution.
2. Install Torch, NumPy, Matplotlib, SciPy, tqdm, and wandb there.
3. Build the ROS 2 workspace using that Python environment.

Example outline:

```bash
conda activate semcomm
source /opt/ros/humble/setup.bash
cd ~/DiskD/RoboticsWorks/scs_robot
colcon build --symlink-install
source install/setup.bash
```

If your ROS 2 distribution is not Humble, replace the setup path accordingly.

## Step 16: Build the Workspace

After refactoring package names and imports:

```bash
cd ~/DiskD/RoboticsWorks/scs_robot
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

If you renamed packages, remove old build artifacts first:

```bash
rm -rf build install log
colcon build --symlink-install
```

Only do the cleanup if you are sure you want a fresh rebuild.

## Step 17: Validate Package Discovery

After the build, check that both packages are registered.

```bash
ros2 pkg list | grep semcomm
```

Or if you kept the old names:

```bash
ros2 pkg list | grep -E 'ml|simulation'
```

Expected result:

- the core package is discoverable
- the runtime package is discoverable

## Step 18: Validate Imports

Check that the runtime node imports the ML package without path injection.

```bash
source install/setup.bash
python -c "from semcomm_core.encoder import SemanticEncoder; print('ok')"
```

If this fails, the packaging or internal imports are still wrong.

## Step 19: Validate Training Entry Points

Check that training is registered as a ROS 2 executable:

```bash
ros2 run semcomm_core train_semcomm --help
```

If your script does not support `--help` yet, at least ensure the entry point starts without import failures.

## Step 20: Validate Runtime Nodes

Check each runtime node individually.

Examples:

```bash
ros2 run semantic_comm_runtime encoder_node
ros2 run semantic_comm_runtime channel_node
ros2 run semantic_comm_runtime decoder_node
ros2 run semantic_comm_runtime nav_node
```

They should start without:

- import errors
- missing checkpoint path errors caused by hardcoded assumptions
- asset path errors

## Step 21: Validate Launch

Run the whole system:

```bash
ros2 launch semantic_comm_runtime sim.launch.py snr:=5.0
```

Validate the following:

1. Gazebo starts.
2. The world loads.
3. The robot spawns.
4. ROS nodes start.
5. Topics are active.

Useful checks:

```bash
ros2 node list
ros2 topic list
ros2 param list
```

## Step 22: Add Tests

At minimum, add these checks:

1. import test for `semcomm_core`
2. import test for runtime nodes
3. launch smoke test
4. one ML unit test for `reparametrize()` or `vib_loss()`

These tests will protect the workspace from future packaging regressions.

## Step 23: Update Documentation

After the migration is complete, update the main documentation so new users only see the new supported flow.

The top-level docs should explain:

1. how to create the Python environment
2. how to build the ROS 2 workspace
3. how to run training through `ros2 run`
4. how to launch simulation through `ros2 launch`
5. where checkpoints and results live

## Common Mistakes To Avoid

### Mistake 1: Making training a ROS node unnecessarily

Training does not need `rclpy` unless it truly interacts with ROS topics.

Keep it as a ROS 2 package console script, not necessarily a ROS node.

### Mistake 2: Keeping hardcoded home-directory paths

This will break on another machine immediately.

Use package-share lookup for assets and parameters for model paths.

### Mistake 3: Depending on source-tree imports

If code only works when run from the repo root, packaging is incomplete.

Everything should work after:

```bash
source install/setup.bash
```

### Mistake 4: Mixing generated artifacts into package source

Keep checkpoints and results under `artifacts/`, not inside installed source packages.

### Mistake 5: Hiding missing dependencies

Document exactly how Torch and other Python packages are installed. ROS 2 will not manage those automatically for you.

## Definition of Done

The migration is complete when all of the following are true:

1. both ML and runtime are normal ROS 2 packages in `src/`
2. no runtime file uses `sys.path.insert(...)`
3. no runtime file uses machine-specific absolute paths
4. runtime nodes import ML modules through installed package imports
5. `colcon build --symlink-install` succeeds from a clean workspace
6. `ros2 run` works for training and runtime entry points
7. `ros2 launch` starts the simulation stack
8. checkpoints and results are externalized under `artifacts/`

## Suggested Execution Order For This Repository

If you want the lowest-risk migration path for this exact repo, do it in this order:

1. rename `src/ml` to `src/semcomm_core`
2. rename the inner Python package from `ml` to `semcomm_core`
3. fix all ML imports
4. rename `src/simulation` to `src/semantic_comm_runtime`
5. fix runtime imports to use `semcomm_core.*`
6. remove `sys.path.insert(...)` from runtime nodes
7. move checkpoint paths to launch parameters
8. update launch file to resolve assets from package share
9. rebuild with `colcon`
10. run node-level validation
11. run full launch validation

## Final Recommendation

Do not try to force every Python file to become a ROS node.

To make the whole project "in ROS 2", the correct engineering interpretation is:

- every executable workflow is delivered through ROS 2 packages
- reusable ML code is installed inside the ROS 2 workspace
- runtime integration uses ROS 2 conventions cleanly

That gives you a full ROS 2 project without turning pure ML utilities into unnecessary ROS wrappers.