#!/usr/bin/env python3
import os
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import PointCloud2

# 优先使用新的 sensor_msgs_py 接口（Humble 及以上），旧接口作为兜底
try:
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    import sensor_msgs.point_cloud2 as pc2


class PCDSaver(Node):
    def __init__(self):
        super().__init__('pcd_saver')
        # 修改为你的实际点云话题
        self.topic = '/livox/lidar'
        # 使用 BEST_EFFORT 保证实时性（也可用 RELIABLE）
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            PointCloud2, self.topic, self.callback, qos)

        # 保存目录（按需修改）
        self.output_dir = '/home/tjurm/LIDAR_lidar/src/gicp/pcd'
        os.makedirs(self.output_dir, exist_ok=True)

        # 监听时长（秒）：到点后停止接收，统一处理一次
        self.listen_duration = 5.0
        # 体素降采样尺寸（米）；0 = 不降采样
        self.voxel_size = 0

        # 累积 5s 内所有帧的点，最后合并成一块点云
        self._points = []
        self._frames = 0
        # 用 Future 通知 main 结束 spin（保存完成后置位）
        self._done = Future()

        # 5s 后触发 _finish_and_save（定时器首次触发正好是 5s 后）
        self._timer = self.create_timer(self.listen_duration, self._finish_and_save)
        self.get_logger().info(
            f'Listening to {self.topic} for {self.listen_duration}s, '
            f'then voxel_down_sample({self.voxel_size}) and save once to {self.output_dir}')

    @staticmethod
    @staticmethod
    def _extract_xyz(msg):
        """从 PointCloud2 中提取 (N, 3) 的 float32 xyz 点坐标。"""
        # 统一拿到「结构化数组」：字段名 x/y/z，元素类型是 record
        pts = None
        if hasattr(pc2, 'read_points_numpy'):
            try:
                pts = pc2.read_points_numpy(
                    msg, field_names=['x', 'y', 'z'], skip_nans=True)
            except Exception:
                pts = None

        if pts is None:
            # 旧接口兜底：read_points 返回生成器，先转 list 再转数组
            pts = pc2.read_points(
                msg, field_names=['x', 'y', 'z'], skip_nans=True)
            pts = np.array(list(pts)) if pts is not None else None

        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 关键：从结构化数组里分别取 x / y / z 三列，再 stack 成 (N,3)
        xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=-1)
        return xyz.astype(np.float32)
    
    def callback(self, msg):
        try:
            points = self._extract_xyz(msg)
            if points.shape[0] == 0:
                return

            # 只累积，不落盘；5s 结束后统一降采样保存一次
            self._points.append(points)
            self._frames += 1
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def _finish_and_save(self):
        """监听计时结束：取消定时器，合并累积点云 → 0.05 降采样 → 保存一次。"""
        try:
            self._timer.cancel()
            if self._frames == 0:
                self.get_logger().warn(
                    f'No points received within {self.listen_duration}s, nothing saved')
                return

            # 合并所有帧为一块点云
            all_pts = np.concatenate(self._points, axis=0).astype(np.float64)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(all_pts)
            raw_n = len(cloud.points)

            # 体素降采样（0 = 跳过）
            if self.voxel_size > 0:
                cloud = cloud.voxel_down_sample(self.voxel_size)

            filename = os.path.join(self.output_dir, 'cloud_once.pcd')
            # write_point_cloud 返回 bool，必须检查是否写入成功
            if o3d.io.write_point_cloud(filename, cloud):
                self.get_logger().info(
                    f'Saved {filename}: {len(cloud.points)} pts '
                    f'(from {raw_n} raw pts / {self._frames} frames, '
                    f'voxel={self.voxel_size})')
            else:
                self.get_logger().error(f'Failed to write {filename}')
        except Exception as e:
            self.get_logger().error(f'Error during save: {e}')
        finally:
            # 通知 main 结束 spin
            if not self._done.done():
                self._done.set_result(True)


def main(args=None):
    rclpy.init(args=args)
    node = PCDSaver()
    try:
        # 一直 spin，直到 5s 后保存完成（_done 被置位）自动退出
        rclpy.spin_until_future_complete(node, node._done)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
