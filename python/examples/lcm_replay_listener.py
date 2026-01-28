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
os.chdir(ROOT_DIR)

from arx5_interface import Arx5CartesianController, EEFState


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
        self._stop_event.clear()
        self._thread = None

    def start(self, data_file: str):
        was_running = self._is_running
        if was_running:
            self.stop()
            self.controller.reset_to_home()
        else:
            self.stop()

        def _run():
            if not os.path.isfile(data_file):
                print(f"Error: File not found: {data_file}")
                return

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


def _decode_payload(data: str) -> Dict[str, Any]:
    """Decode LCM payload.

    Supported formats:
    - Plain string: "teach_traj" or "stop"
    """
    if isinstance(data, bytes):
        text = data.decode("utf-8").strip()
    else:
        text = str(data).strip()

    print("received cmd:", text)

    if text.lower() == "stop":
        return {"cmd": "stop"}

    return {"data_file": text}


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option("--address", default="239.255.76.67", help="LCM multicast address")
@click.option("--port", default=7667, help="LCM multicast port")
@click.option("--ttl", default=1, help="LCM multicast TTL")
@click.option("--channel", default="ARX5_REPLAY", help="LCM channel to listen for replay commands")
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

    def handler(_, data):
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
        print("Received replay command for:", data_file)
        data_file = os.path.join(os.path.dirname(ROOT_DIR), "offline_traj", f"{data_file}.npy")
        runner.start(data_file)

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
