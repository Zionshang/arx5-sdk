import argparse
import os
import sys

import numpy as np
import time

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


def print_gain(prefix: str, gain: dict) -> None:
    kp = gain["kp"]
    kd = gain["kd"]
    print(
        f"{prefix} kp={np.array2string(kp, precision=4)}, "
        f"kd={np.array2string(kd, precision=4)}, "
        f"gripper_kp={gain['gripper_kp']:.4f}, "
        f"gripper_kd={gain['gripper_kd']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Arx5LcmClient against a real robot.")
    parser.add_argument("--url", default="", help="LCM URL, e.g. udpm://239.255.76.67:7667?ttl=1")
    parser.add_argument("--address", default="239.255.76.67", help="LCM multicast address")
    parser.add_argument("--port", type=int, default=7667, help="LCM multicast port")
    parser.add_argument("--ttl", type=int, default=1, help="LCM multicast TTL")
    parser.add_argument("--auto", action="store_true", help="Run without interactive confirmations")
    args = parser.parse_args()

    print("Connecting to LCM server...")
    client = Arx5LcmClient(url=args.url, address=args.address, port=args.port, ttl=args.ttl)

    print("\n[1/7] GET_STATE")
    state = client.get_state()
    print_state("Current", state)
    time.sleep(1.0)

    print("\n[2/7] GET_GAIN + SET_GAIN (round trip)")
    gain = client.get_gain()
    print_gain("Current gain", gain)
    client.set_gain(gain)
    print("Gain round trip OK")
    time.sleep(1.0)

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
        target_pose[2] += 0.1
        client.set_ee_pose(target_pose, gripper_pos=0.4, preview_time=2.0)
        state = client.get_state()
        time.sleep(2.5)
        print_state("After move", state)
    else:
        print("Skipped")

    print("\n[5/6] SET_EE_POSE without gripper_pos and preview_time")
    if ask_yes_no("Send same pose without gripper_pos?", args.auto):
        state = client.get_state()
        target_pose = state["ee_pose"].copy()
        target_pose[2] -= 0.05
        client.set_ee_pose(target_pose, gripper_pos=None, preview_time=None)
        time.sleep(1.0)
        print_state("After move", state)
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
