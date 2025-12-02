import lcm
import struct
import threading
import time

def listen_status(lc):
    def handler(channel, data):
        try:
            status, obj_id = struct.unpack("ii", data)
            res = "Success" if status == 0 else "Fail"
            print(f"\n[LCM RX] {channel}: Status={res}({status}), ObjID={obj_id}")
            print("Command > ", end="", flush=True)
        except Exception as e:
            print(f"[LCM Error] Decode failed: {e}")

    lc.subscribe("ARM_STATUS", handler)
    while True:
        try:
            lc.handle()
        except KeyboardInterrupt:
            break

def main():
    lc = lcm.LCM()
    
    # Start listener thread
    t = threading.Thread(target=listen_status, args=(lc,), daemon=True)
    t.start()

    print("=== LCM Control Tester ===")
    print("Commands: 0 = Grasp, 1 = Place, q = Quit")
    
    while True:
        try:
            val = input("Command > ")
            if val.lower() == 'q':
                break
            
            cmd = int(val)
            if cmd in [0, 1]:
                msg = struct.pack("i", cmd)
                lc.publish("ARM_CMD", msg)
                print(f"[LCM TX] Sent ARM_CMD: {cmd}")
            else:
                print("Invalid command. Enter 0 or 1.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
