#!/usr/bin/env python3
"""自测用：把 cloud_once.pcd 按给定位姿的逆变换后，当作 /livox/lidar 发出去。

仿真模型：地图 = 录制时刻传感器坐标系 F0 下的点集。
当前传感器位于 F0 下的位姿 T（即 map->livox）。
则物理点在当前传感器坐标系下的观测为 p_F1 = T^-1 * p_F0。
GICP 用初值 identity 把 p_F1 对齐到地图，应当解出 T_corr ~= T。
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


def load_pcd(path):
    d = open(path, 'rb').read()
    i = d.index(b'DATA binary') + len(b'DATA binary\n')
    hdr = d[:i].decode('ascii', 'ignore')
    n = int([l for l in hdr.splitlines() if l.startswith('POINTS')][0].split()[1])
    return np.frombuffer(d[i:i + n * 12], dtype=np.float32).reshape(n, 3).copy()


def main():
    tx, ty, tz, yaw = (float(v) for v in sys.argv[1:5])
    rclpy.init()
    node = Node('fake_scan_publisher')

    pts = load_pcd('/home/tjurm/LIDAR_lidar/src/gicp/pcd/cloud_once.pcd')

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = np.array([tx, ty, tz], dtype=np.float64)
    # p_F1 = R^T (p - t) = T^-1 p
    pts_F1 = (pts.astype(np.float64) - t) @ R

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    pub = node.create_publisher(PointCloud2, '/livox/lidar', qos)

    msg = PointCloud2()
    msg.header.frame_id = 'livox_frame'
    msg.height = 1
    msg.width = len(pts_F1)
    msg.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(pts_F1)
    msg.is_dense = True
    msg.data = pts_F1.astype(np.float32).tobytes()

    node.get_logger().info(
        f'publishing ground-truth pose T = t=({tx:+.3f},{ty:+.3f},{tz:+.3f}) yaw={yaw:+.4f} rad '
        f'({np.degrees(yaw):+.2f} deg), {len(pts_F1)} pts, frame=livox_frame')

    end = time.time() + float(sys.argv[5]) if len(sys.argv) > 5 else time.time() + 20.0
    while time.time() < end and rclpy.ok():
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        time.sleep(0.1)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
