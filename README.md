# mujoco-ocp

Minimal MuJoCo OCP experiments for building and testing an optimal control pipeline step by step.

Initial models:
- Unitree G1
- Robotiq 2F85 v4

Current focus:
1. Load and inspect MuJoCo models.
2. Attach Robotiq 2F85 v4 to Unitree G1.
3. Build simple kinematics and dynamics checks.
4. Add OCP components gradually.

## Environment

Tested with:

- Python 3.11.15
- MuJoCo 3.6.0
- NumPy 2.4.3
- CasADi 3.7.2
- Pinocchio 4.0.0

## Install

From the repository root:

```bash
pip install -e .