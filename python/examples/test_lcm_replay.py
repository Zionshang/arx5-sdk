import time
import lcm


def main():
    address = "239.255.76.67"
    port = 7667
    ttl = 1
    channel = "ARX5_REPLAY"

    # (message, sleep_seconds) sequence
    sequence = [
        ("teach_traj", 10.0),
        ("teach_traj2", 10.0),
        ("teach_traj3", 10.0),
        ("stop", 0.0),
    ]

    lcm_url = f"udpm://{address}:{port}?ttl={ttl}"
    lc = lcm.LCM(lcm_url)

    for message, sleep_seconds in sequence:
        payload = message.strip()
        if not payload:
            raise SystemExit("message cannot be empty")
        lc.publish(channel, payload.encode("utf-8"))
        print(f"Sent to {channel}: {payload}")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
