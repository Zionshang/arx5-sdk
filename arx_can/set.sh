#!/bin/bash
source ~/.bashrc

sudo cp arx_can.rules /etc/udev/rules.d/
echo "Copied CAN rules to /etc/udev/rules.d/"

sudo chmod +x /etc/udev/rules.d/arx_can.rules
echo "Set execute permissions on CAN rules."

sudo udevadm control --reload-rules && sudo udevadm trigger
echo "Reloaded udev rules."

echo "Finished."


