# SCS Robot Architecture Blueprint

## Goal
Compartmentalize the project into clear domains so training, ROS runtime, simulation assets, and generated artifacts are isolated.

## Recommended Top-Level Layout

scs_robot/
  apps/
    train/
      run_train.py
    eval/
      run_eval.py
  packages/
    semcomm_core/
      pyproject.toml
      src/semcomm_core/
        channel/
        data/
        evaluation/
        loss/
        models/
        train/
        config/
  ros2/
    ws/
      src/
        semantic_comm/
          package.xml
          setup.py
          setup.cfg
          launch/
          semantic_comm/
          assets/
            robot/
            world/
  infra/
    docker/
      Dockerfile
      docker-compose.yml
    requirements/
      base.txt
      training.txt
      ros2.txt
  artifacts/
    checkpoints/
    results/
    wandb/
  docs/
    architecture.md
    runbooks.md
  tests/
    unit/
    integration/

## Why This Works
- apps: Only executable entry points.
- packages: Reusable Python package code with no runtime side effects.
- ros2: Runtime and simulator concerns isolated from ML training concerns.
- infra: Build and environment files isolated from code.
- artifacts: All generated outputs grouped and easy to clean.
- docs/tests: Non-runtime assets clearly separated.

## Import Rules
- All training and evaluation imports come from semcomm_core.*
- ROS nodes import semcomm_core via installed package, never via sys.path append hacks.
- Entry-point scripts in apps/ should remain thin wrappers around library functions.

## Current-to-Target Mapping (Incremental)
1. src/basecode/* -> packages/semcomm_core/src/semcomm_core/*
2. src/run_train.py -> apps/train/run_train.py
3. src/ros2_ws/* -> ros2/ws/*
4. src/Dockerfile + src/docker-compose.yml -> infra/docker/*
5. results/ + wandb/ + checkpoints -> artifacts/*

## Naming Rules
- Keep only one canonical name for each asset folder:
  - robot
  - world
- Avoid parallel names like urdf and robot, or world and worlds, unless there is a strict format reason.

## Config Strategy
- packages/semcomm_core/src/semcomm_core/config/
  - train.yaml
  - eval.yaml
  - model.yaml
- ros2/ws/src/semantic_comm/config/
  - runtime.yaml
- Avoid hard-coded absolute paths; resolve paths relative to package/share directories.

## Minimum Git Ignore Policy
- artifacts/checkpoints/*
- artifacts/results/*
- artifacts/wandb/*
- Keep only curated, small benchmark outputs if needed.

## Refactor Order (Low-Risk)
1. Create semcomm_core package and move basecode modules first.
2. Update imports in trainer and run scripts.
3. Fix ROS package asset folder naming and launch path resolution.
4. Move Docker and dependency files to infra.
5. Move generated outputs to artifacts and update compose mounts.

## Definition of Done
- No import from src.* remains.
- No hardcoded machine path remains.
- ROS launch resolves assets from package paths.
- One command each for train, eval, and simulation from docs/runbooks.
