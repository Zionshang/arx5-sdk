import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from communication.lcm.lcm_client import Arx5LcmClient


def ask_yes_no(prompt: str, auto: bool) -> bool:
    if auto:
        return True
    while True:
        resp = input(f"{prompt} [y/N]: ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no", ""):
            return False


def print_state(prefix: str, state: dict) -> None:
    ee_pose = state["ee_pose"]
    print(
        f"{prefix} timestamp={state['timestamp']:.3f}, "
        f"ee_pose={np.array2string(ee_pose, precision=4)}, "
        f"gripper_pos={state['gripper_pos']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Arx5LcmClient against a real robot.")
    parser.add_argument("--url", default="", help="LCM URL, e.g. udpm://239.255.76.67:7667?ttl=1")
    parser.add_argument("--address", default="239.255.76.67", help="LCM multicast address")
    parser.add_argument("--port", type=int, default=7667, help="LCM multicast port")
    parser.add_argument("--ttl", type=int, default=1, help="LCM multicast TTL")
    parser.add_argument("--auto", action="store_true", help="Run without interactive confirmations")
    parser.add_argument("--move-step", type=float, default=0.01, help="Small EE x-axis move in meters")
    parser.add_argument("--preview-time", type=float, default=0.1, help="Preview time in seconds")
    args = parser.parse_args()

    print("Connecting to LCM server...")
    client = Arx5LcmClient(url=args.url, address=args.address, port=args.port, ttl=args.ttl)

    print("\n[1/7] GET_STATE")
    state = client.get_state()
    print_state("Current", state)

    print("\n[2/7] GET_GAIN + SET_GAIN (round trip)")
    gain = client.get_gain()
    client.set_gain(gain)
    print("Gain round trip OK")

    print("\n[3/7] RESET_TO_HOME (optional)")
    if ask_yes_no("Reset robot to home position?", args.auto):
        client.reset_to_home()
        state = client.get_state()
        print_state("After reset", state)
    else:
        print("Skipped")

    print("\n[4/7] SET_EE_POSE with preview_time (small delta)")
    if ask_yes_no("Send small EE pose delta on +X?", args.auto):
        state = client.get_state()
        target_pose = state["ee_pose"].copy()
        target_pose[0] += args.move_step
        client.set_ee_pose(target_pose, gripper_pos=None, preview_time=args.preview_time)
        state = client.get_state()
        print_state("After move", state)
    else:
        print("Skipped")

    print("\n[5/6] SET_EE_POSE without gripper_pos (should keep gripper)")
    if ask_yes_no("Send same pose without gripper_pos?", args.auto):
        state = client.get_state()
        before_gripper = float(state["gripper_pos"])
        client.set_ee_pose(state["ee_pose"], gripper_pos=None, preview_time=None)
        state = client.get_state()
        after_gripper = float(state["gripper_pos"])
        print_state("After gripper hold", state)
        if abs(after_gripper - before_gripper) < 1e-4:
            print("Gripper position unchanged (OK)")
        else:
            print(f"Gripper position changed: {before_gripper:.4f} -> {after_gripper:.4f}")
    else:
        print("Skipped")

    print("\n[6/6] SET_TO_DAMPING (optional)")
    if ask_yes_no("Set robot to damping mode?", args.auto):
        client.set_to_damping()
        print("Damping mode set")
    else:
        print("Skipped")

    print("\nDone.")


if __name__ == "__main__":
    main()
