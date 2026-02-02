import os
import sys
import threading
import time
from typing import Any, Dict

import lcm
import numpy as np
import click

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "communication", "lcm"))
os.chdir(ROOT_DIR)

from arx5_interface import Arx5CartesianController, EEFState
from communication.lcm.msg.TaskGroupData import TaskGroupData


class ReplayRunner:
    def __init__(self, controller: Arx5CartesianController):
        self.controller = controller
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        self._is_running = False
        self._stop_event.clear()
        self._thread = None

    def start(self, data_file: str) -> bool:
        if self._is_running or (self._thread and self._thread.is_alive()):
            print("Replay already running; ignoring new command.")
            return False
        if not os.path.isfile(data_file):
            print(f"Error: File not found: {data_file}")
            return False
        self.stop()

        def _run():
            self._is_running = True
            traj = np.load(data_file, allow_pickle=True)
            if len(traj) == 0:
                self._is_running = False
                print(f"Empty trajectory file: {data_file}")
                return
            controller_config = self.controller.get_controller_config()
            for pose_dict in traj:
                if self._stop_event.is_set():
                    self._is_running = False
                    return
                pose_6d = pose_dict["pose_6d"]
                gripper_pos = pose_dict["gripper_pos"]
                self.controller.set_eef_cmd(EEFState(pose_6d, gripper_pos))
                time.sleep(controller_config.controller_dt)
            self.controller.reset_to_home()
            self._is_running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True


def _decode_payload(data) -> Dict[str, Any]:
    """Decode LCM payload.

    Supported formats:
    - Plain string: "teach_traj" or "stop"
    """
    msg = TaskGroupData.decode(data)
    text = msg.task_group.strip()
    print("received task_group:", text)

    if text.lower() == "stop":
        return {"cmd": "stop"}

    return {"data_file": text}


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option("--address", default="239.255.76.67", help="LCM multicast address")
@click.option("--port", default=20000, help="LCM multicast port")
@click.option("--ttl", default=1, help="LCM multicast TTL")
@click.option("--channel", default="mc_arm_command_desired_galileo", help="LCM channel to listen for replay commands")
def main(
    model: str,
    interface: str,
    address: str,
    port: int,
    ttl: int,
    channel: str,
):
    controller = Arx5CartesianController(model, interface)
    controller.reset_to_home()

    runner = ReplayRunner(controller)

    lcm_url = f"udpm://{address}:{port}?ttl={ttl}"
    lc = lcm.LCM(lcm_url)
    last_command: str | None = None

    def handler(_, data):
        nonlocal last_command
        payload = _decode_payload(data)
        if payload.get("cmd") == "stop":
            print("Replay stop requested. Exiting...")
            runner.stop()
            controller.reset_to_home()
            sys.exit(0)

        data_file = payload.get("data_file")
        if not data_file:
            print("Replay ignored: missing data_file in payload.")
            return
        if last_command == data_file:
            print("Replay ignored: same command as last executed:", data_file)
            return
        print("Received replay command for:", data_file)
        data_file = os.path.join(os.path.dirname(ROOT_DIR), "offline_traj", f"{data_file}.npy")
        if runner.start(data_file):
            last_command = payload.get("data_file")

    lc.subscribe(channel, handler)
    print(f"Listening for replay commands on {channel} via {lcm_url}...")

    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        print("Stopping replay listener.")
        runner.stop()


if __name__ == "__main__":
    main()
