import time
import lcm

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "communication", "lcm"))
from communication.lcm.msg.TaskGroupData import TaskGroupData


def main():
    address = "239.255.76.67"
    port = 20000
    ttl = 1
    channel = "mc_arm_command_desired_galileo"
    period = 0.01  # seconds between messages

    # (message, duration_seconds) sequence
    sequence = [
        ("0", 15.0),
        ("1", 15.0),
        ("2", 15.0),
    ]

    lcm_url = f"udpm://{address}:{port}?ttl={ttl}"
    lc = lcm.LCM(lcm_url)

    for message, duration_seconds in sequence:
        task_group = message.strip()
        end_time = time.monotonic() + duration_seconds
        while time.monotonic() < end_time:
            msg = TaskGroupData()
            msg.task_group = task_group
            lc.publish(channel, msg.encode())
            print(f"Sent to {channel}: {task_group}")
            time.sleep(period)


if __name__ == "__main__":
    main()
