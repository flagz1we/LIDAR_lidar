#!/usr/bin/env python3
"""列出 /tf 与 /tf_static 上所有边及其数值，用来验证节点到底发没发 TF。"""
import sys
import time

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    rclpy.init()
    n = Node('tf_probe')
    seen = {}

    def cb(m):
        for t in m.transforms:
            seen[(t.header.frame_id, t.child_frame_id)] = t

    n.create_subscription(TFMessage, '/tf', cb, 50)
    n.create_subscription(TFMessage, '/tf_static', cb, 50)
    end = time.time() + dur
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.2)

    print(f'TF edges seen in {dur:.0f}s: {len(seen)}')
    for (p, c), t in sorted(seen.items()):
        tr, q = t.transform.translation, t.transform.rotation
        print(f'  {p} -> {c} | xyz=({tr.x:+.3f},{tr.y:+.3f},{tr.z:+.3f}) '
              f'q=({q.x:+.3f},{q.y:+.3f},{q.z:+.3f},{q.w:+.3f})')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
