#!/bin/bash
source ~/.bashrc

# 遍历每一个匹配的设备
for device in /dev/ttyACM*; do
    # 检查文件是否存在（防止没有设备时报错）
    [ -e "$device" ] || continue
    
    echo "Checking device: $device"
    # 对每个设备单独执行命令
    udevadm info -a -n "$device" | grep serial
done