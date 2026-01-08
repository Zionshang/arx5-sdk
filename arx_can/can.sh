#!/bin/bash

source ~/.bashrc

CAN_DEVICE="/dev/arxcan0"
CAN_INTERFACE="can0"

echo "启动 slcand..."
sudo slcand -o -f -s8 $CAN_DEVICE $CAN_INTERFACE
if [ $? -ne 0 ]; then
    echo "slcand 启动失败"
    exit 1
fi

echo "配置 $CAN_INTERFACE 接口..."
sudo ifconfig $CAN_INTERFACE up

if [ $? -ne 0 ]; then
    echo "启动 $CAN_INTERFACE 接口失败"
    exit 1
fi

echo "$CAN_INTERFACE 启动成功"
