#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <mutex>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

class CloudSubtractor : public rclcpp::Node
{
public:
  CloudSubtractor()
  : Node("cloud_subtractor")
  {
    // ---------- 参数 ----------
    predefined_path_  = this->declare_parameter<std::string>("predefined_pcd", "");
    threshold_        = this->declare_parameter<double>("distance_threshold", 0.05);
    input_topic_      = this->declare_parameter<std::string>("input_topic",  "input_cloud");
    output_topic_     = this->declare_parameter<std::string>("output_topic", "remaining_cloud");
    removed_topic_    = this->declare_parameter<std::string>("removed_topic","removed_cloud");
    predefined_frame_ = this->declare_parameter<std::string>("predefined_frame", "");
    process_rate_     = this->declare_parameter<double>("process_rate_hz", 10.0);

    if (predefined_path_.empty())
      throw std::runtime_error("param 'predefined_pcd' is required");


    // ---------- TF ----------
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ---------- 订阅 / 发布 ----------
    pub_         = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_,  rclcpp::SensorDataQoS());
    pub_removed_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(removed_topic_, rclcpp::SensorDataQoS());
    pub_centroids_ = this->create_publisher<geometry_msgs::msg::PoseArray>("cluster_centroids", rclcpp::QoS(10));
    pub_centroids_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cluster_centroids_marker", rclcpp::QoS(10));
    sub_         = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CloudSubtractor::cloudCallback, this, std::placeholders::_1));

    // ---------- 定时器 ----------
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(0.1, process_rate_)),
      std::bind(&CloudSubtractor::timerCallback, this));

    RCLCPP_INFO(get_logger(), "sub=%s pub=%s/%s rate=%.1fHz",
                input_topic_.c_str(), output_topic_.c_str(),
                removed_topic_.c_str(), process_rate_);
  }

private:
  // 回调只缓存最新帧
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_msg_ = msg;
    if(game_state_ == 3 && !get_predefined_)
    {
        auto raw = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(*msg, *raw);

        auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(raw);
        vg.setLeafSize(0.05f, 0.05f, 0.05f);   // 按你的场景调
        vg.filter(*filtered);
        predefined_ = filtered;
        get_predefined_ = true;
    }
  }

  // 定时回调：取帧 -> 查 TF -> 变换预先点云 -> 建 KD-Tree -> 剔除 -> 发布
  void timerCallback()
  {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr msg;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!latest_msg_) return;
      if (last_stamp_set_ && last_stamp_ == latest_msg_->header.stamp) return;  // 同一帧跳过
      msg = latest_msg_;
      last_stamp_ = msg->header.stamp;
      last_stamp_set_ = true;
    }
    processCloud(*msg);
  }

  void processCloud(const sensor_msgs::msg::PointCloud2 & msg)
  {
    // 1) ROS -> PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr current(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(msg, *current);
    if (current->empty()) return;

    // 2) 查询 TF：把预先点云变到当前点云 frame
    const std::string target = "odom";
    const std::string source = "map";
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    try {
      auto tf_msg = tf_buffer_->lookupTransform(
        target, source,
        rclcpp::Time(msg.header.stamp),
        tf2::durationFromSec(0.1));
      T = tf2::transformToEigen(tf_msg).matrix().cast<float>();
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "TF %s -> %s failed: %s", source.c_str(), target.c_str(), ex.what());
      return;
    }

    // 3) 变换预先点云 + 建 KD-Tree
    pcl::PointCloud<pcl::PointXYZ>::Ptr predefined_t(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*predefined_, *predefined_t, T);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(predefined_t);

    // 4) 最近邻 + 阈值剔除
    pcl::PointIndices::Ptr to_remove(new pcl::PointIndices);
    std::vector<int>   nn_idx(1);
    std::vector<float> nn_sq(1);
    const float th2 = static_cast<float>(threshold_ * threshold_);  // PCL 返回平方距离

    for (size_t i = 0; i < current->size(); ++i) {
      if (kdtree.nearestKSearch(current->points[i], 1, nn_idx, nn_sq) > 0 &&
          nn_sq[0] < th2) {
        to_remove->indices.push_back(static_cast<int>(i));
      }
    }

    // 5) 提取
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(current);
    extract.setIndices(to_remove);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(true);
    extract.filter(*remaining);

    pcl::PointCloud<pcl::PointXYZ>::Ptr removed(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(false);
    extract.filter(*removed);

    // 6) 发布
    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(*removed, out);
    out.header = msg.header;
    pub_removed_->publish(out);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
    "current=%zu removed=%zu remaining=%zu",
    current->size(), removed->size(), remaining->size());

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(remaining);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.05);   // 邻域半径，m
    ec.setMinClusterSize(30);       // 少于这个点数的簇丢掉
    ec.setMaxClusterSize(10000);    // 多于这个点数的簇丢掉（可选）
    ec.setSearchMethod(tree);
    ec.setInputCloud(remaining);
    ec.extract(cluster_indices);

    // 每个 cluster_indices[i] 是一簇的点索引
    std::vector<Eigen::Vector4f> centroids;
    centroids.reserve(cluster_indices.size());
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        const auto & ci = cluster_indices[i];
        if (ci.indices.empty()) continue;
        Eigen::Vector4f c;
        pcl::compute3DCentroid(*remaining, ci.indices, c);
        centroids.push_back(c);

        RCLCPP_INFO(get_logger(), "cluster %zu: n=%zu centroid=(%.2f, %.2f, %.2f)",
                i, ci.indices.size(), c[0], c[1], c[2]);
    }
    geometry_msgs::msg::PoseArray pose_arr;
    pose_arr.header = msg.header;
    pose_arr.poses.reserve(centroids.size());
    for (const auto & c : centroids) {
        geometry_msgs::msg::Pose p;
        p.position.x = c[0];
        p.position.y = c[1];
        p.position.z = c[2];
        p.orientation.w = 1.0;   // 只用位置，朝向给单位四元数
        pose_arr.poses.push_back(p);
    }
    pub_centroids_->publish(pose_arr);
    sensor_msgs::msg::PointCloud2 remaining_msg;
    pcl::toROSMsg(*remaining, remaining_msg);
    remaining_msg.header = msg.header;
    pub_->publish(remaining_msg);

    // 3) 可视化话题：MarkerArray
    visualization_msgs::msg::MarkerArray marker_arr;

    // 3a) 这一帧簇数比上一帧少时，把多出来的旧 id 显式 DELETE，否则 RViz 里会残留
    for (size_t i = centroids.size(); i < last_marker_count_; ++i) {
        visualization_msgs::msg::Marker del;
        del.header = msg.header;
        del.ns = "cluster_centroids";
        del.id = static_cast<int>(i);
        del.action = visualization_msgs::msg::Marker::DELETE;
        marker_arr.markers.push_back(del);
    }

    // 3b) 每个质心一个球
    for (size_t i = 0; i < centroids.size(); ++i) {
        visualization_msgs::msg::Marker m;
        m.header = msg.header;
        m.ns = "cluster_centroids";
        m.id = static_cast<int>(i);
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = centroids[i][0];
        m.pose.position.y = centroids[i][1];
        m.pose.position.z = centroids[i][2];
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.2;
        m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 1.0f;
        // 额外保险：0.3s 自动过期，即使 DELETE 漏了也不会永久残留
        m.lifetime = rclcpp::Duration::from_seconds(0.3);
        marker_arr.markers.push_back(m);
    }
    last_marker_count_ = centroids.size();
    pub_centroids_marker_->publish(marker_arr);
  }

  // ---------- 成员 ----------
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr    pub_centroids_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_, pub_removed_;
  rclcpp::TimerBase::SharedPtr timer_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr predefined_;

  std::mutex mutex_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr latest_msg_;
  builtin_interfaces::msg::Time last_stamp_;
  bool last_stamp_set_ = false;

  std::string predefined_path_, predefined_frame_;
  std::string input_topic_, output_topic_, removed_topic_;
  double threshold_    = 0.05;
  double process_rate_ = 10.0;

  bool get_predefined_ = false;
  int game_state_ = 3;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_centroids_marker_;
  size_t last_marker_count_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<CloudSubtractor>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("cloud_subtractor"), "fatal: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}