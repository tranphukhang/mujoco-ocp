import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


FPS = 30
FRAME_TIME = 1.0 / FPS


def set_keyframe_if_exists(model, data, key_name: str):
    """Set model state from keyframe if key_name exists."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)

    if key_id == -1:
        print(f"[WARN] Keyframe '{key_name}' not found. Use default qpos.")
        return

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    print(f"[INFO] Loaded keyframe: {key_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=str,
        default="assets/g1_gripper/G1_with_gripper.xml",
        help="Path to MuJoCo XML file.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Optional keyframe name, for example: home or stand.",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Only view model without stepping simulation.",
    )

    args = parser.parse_args()

    xml_path = Path(args.xml).resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"[INFO] Loading model: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("[INFO] Model loaded successfully")
    print(f"[INFO] nq = {model.nq}")
    print(f"[INFO] nv = {model.nv}")
    print(f"[INFO] nu = {model.nu}")
    print(f"[INFO] timestep = {model.opt.timestep}")
    print("\n[JOINTS]")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(i, name)

    print("\n[ACTUATORS]")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(i, name)

    if args.key is not None:
        set_keyframe_if_exists(model, data, args.key)
    else:
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] Viewer started")
        print(f"[INFO] Rendering at {FPS} FPS")

        while viewer.is_running():
            frame_start = time.time()

            if not args.pause:
                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - frame_start
            sleep_time = max(0.0, FRAME_TIME - elapsed)
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()