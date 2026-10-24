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
#include <chrono>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <limits>

class FrameConsistencyFilter {
public:
    enum class State { Tentative, Confirmed, Lost };

    struct Params {
        float base_radius      = 2.0f;   // 基础匹配半径（m）
        float radius_per_m     = 0.05f;  // 每米额外半径
        int   confirm_hits     = 3;      // Tentative→Confirmed 所需连续命中帧数
        int   tentative_max_miss = 2;    // Tentative 连续失配超过此值直接删除
        int   lost_max_miss    = 8;      // Confirmed→Lost 后连续失配超过此值删除
    };
    Params p;

    // 返回已确认质心；id 与 out 一一对应（供 Marker 稳定使用）
    std::vector<Eigen::Vector4f> update(
        const std::vector<Eigen::Vector4f> & dets,
        std::vector<int> * ids_out = nullptr)
    {
        const size_t old_count = tracks_.size();
        std::vector<bool> track_matched(old_count, false);
        std::vector<bool> det_matched(dets.size(), false);

        // 1) 匹配：所有状态（Tentative / Confirmed / Lost）都参与
        for (size_t i = 0; i < dets.size(); ++i) {
            Eigen::Vector3f d(dets[i][0], dets[i][1], dets[i][2]);
            const float r = p.base_radius + p.radius_per_m * d.norm();
            float best = std::numeric_limits<float>::max();
            int best_j = -1;
            for (size_t j = 0; j < old_count; ++j) {
                if (track_matched[j]) continue;
                float dist = (tracks_[j].pos - d).norm();
                if (dist < r && dist < best) { best = dist; best_j = (int)j; }
            }
            if (best_j >= 0) {
                tracks_[best_j].pos = d;
                track_matched[best_j] = true;
                det_matched[i] = true;
            }
        }

        // 2) 状态转移
        std::vector<Track> kept;
        kept.reserve(old_count + dets.size());
        for (size_t j = 0; j < old_count; ++j) {
            Track & t = tracks_[j];
            if (track_matched[j]) {
                t.miss = 0;
                t.hit += 1;
                switch (t.state) {
                    case State::Tentative:
                        if (t.hit >= p.confirm_hits) t.state = State::Confirmed;
                        break;
                    case State::Confirmed:
                        break;
                    case State::Lost:
                        // 复活：直接回到 Confirmed（不需要重新试
                        // 用期，身份已建立）
                        t.state = State::Confirmed;
                        break;
                }
                kept.push_back(t);
            } else {
                t.miss += 1;
                switch (t.state) {
                    case State::Tentative:
                        if (t.miss <= p.tentative_max_miss) kept.push_back(t);
                        // 否则丢弃（不加入 kept）
                        break;
                    case State::Confirmed:
                        t.state = State::Lost;
                        if (t.miss <= p.lost_max_miss) kept.push_back(t);
                        break;
                    case State::Lost:
                        if (t.miss <= p.lost_max_miss) kept.push_back(t);
                        // 否则丢弃
                        break;
                }
            }
        }
        tracks_.swap(kept);

        // 3) 未匹配的检测：全部进入 Tentative
        for (size_t i = 0; i < dets.size(); ++i) {
            if (det_matched[i]) continue;
            Track t;
            t.pos   = Eigen::Vector3f(dets[i][0], dets[i][1], dets[i][2]);
            t.state = State::Tentative;
            t.hit   = 1;
            t.miss  = 0;
            t.id    = next_id_++;
            tracks_.push_back(t);
        }

        // 4) 只输出 Confirmed
        std::vector<Eigen::Vector4f> out;
        out.reserve(tracks_.size());
        if (ids_out) ids_out->clear();
        for (const auto & t : tracks_) {
            if (t.state == State::Confirmed) {
                out.emplace_back(t.pos.x(), t.pos.y(), t.pos.z(), 1.0f);
                if (ids_out) ids_out->push_back(t.id);
            }
        }
        return out;
    }

    void reset() { tracks_.clear(); next_id_ = 0; }

private:
    struct Track {
        Eigen::Vector3f pos;
        State state = State::Tentative;
        int   hit   = 1;   // 连续命中计数（匹配清零 miss、hit++）
        int   miss  = 0;   // 连续失配计数（匹配清零）
        int   id    = -1;
    };
    std::vector<Track> tracks_;
    int next_id_ = 0;
};

class CloudSubtractor : public rclcpp::Node
{
public:
  CloudSubtractor()
  : Node("cloud_subtractor")
  {
    // ---------- 参数 ----------
    predefined_path_    = this->declare_parameter<std::string>("predefined_pcd", "/home/tjurm/LIDAR_lidar/src/gicp/pcd/cloud_once.pcd");
    threshold_          = this->declare_parameter<double>("distance_threshold", 0.1);
    input_topic_        = this->declare_parameter<std::string>("input_topic", "/livox/lidar");
    output_topic_       = this->declare_parameter<std::string>("output_topic", "remaining_cloud");
    removed_topic_      = this->declare_parameter<std::string>("removed_topic", "removed_cloud");
    predefined_frame_   = this->declare_parameter<std::string>("predefined_frame", "livox_frame");
    process_rate_       = this->declare_parameter<double>("process_rate_hz", 10.0);
    near_distance_      = this->declare_parameter<double>("near_distance", 8.0);
    near_accum_frames_  = this->declare_parameter<int>("near_accum_frames", 2);
    far_accum_frames_   = this->declare_parameter<int>("far_accum_frames", 5);

    // ---------- TF ----------
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // ---------- 订阅 / 发布 ----------
    pub_                  = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::QoS(10));
    pub_removed_          = this->create_publisher<sensor_msgs::msg::PointCloud2>(removed_topic_, rclcpp::QoS(10));
    pub_centroids_        = this->create_publisher<geometry_msgs::msg::PoseArray>("cluster_centroids", rclcpp::QoS(10));
    pub_centroids_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("cluster_centroids_marker", rclcpp::QoS(10));

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CloudSubtractor::cloudCallback, this, std::placeholders::_1));

    // ---------- 定时器 ----------
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(0.1, process_rate_)),
      std::bind(&CloudSubtractor::timerCallback, this));

    // ---------- 加载先验点云 ----------
    if (!predefined_path_.empty()) {
      auto raw = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      if (pcl::io::loadPCDFile<pcl::PointXYZ>(predefined_path_, *raw) == 0 && !raw->empty()) {
        auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(raw);
        vg.setLeafSize(0.02f, 0.02f, 0.02f);
        vg.filter(*filtered);
        predefined_ = filtered;
        RCLCPP_INFO(get_logger(), "Loaded predefined cloud: %zu -> %zu pts from %s (frame=%s)",
                    raw->size(), predefined_->size(),
                    predefined_path_.c_str(), predefined_frame_.c_str());
      } else {
        RCLCPP_ERROR(get_logger(), "Failed to load predefined PCD: %s", predefined_path_.c_str());
      }
    } else {
      RCLCPP_WARN(get_logger(), "predefined_pcd empty, skip loading");
    }

    RCLCPP_INFO(get_logger(),
                "sub=%s pub=%s/%s rate=%.1fHz near_dist=%.1fm near_accum=%d far_accum=%d",
                input_topic_.c_str(), output_topic_.c_str(), removed_topic_.c_str(),
                process_rate_, near_distance_, near_accum_frames_, far_accum_frames_);
  }

private:
  struct CloudEntry {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    std_msgs::msg::Header header;
  };

  // 回调：按距离分箱并累积到 near/far 滑动窗口
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    auto raw = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *raw);
    if (raw->empty()) return;

    auto near_part = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto far_part  = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    const double near_th2 = near_distance_ * near_distance_;
    for (const auto & p : raw->points) {
      const double d2 = static_cast<double>(p.x) * p.x
                      + static_cast<double>(p.y) * p.y
                      + static_cast<double>(p.z) * p.z;
      if (d2 < near_th2) near_part->push_back(p);
      else               far_part->push_back(p);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    near_buffer_.push_back({near_part, msg->header});
    while (near_buffer_.size() > static_cast<size_t>(std::max(1, near_accum_frames_))) {
      near_buffer_.pop_front();
    }
    far_buffer_.push_back({far_part, msg->header});
    while (far_buffer_.size() > static_cast<size_t>(std::max(1, far_accum_frames_))) {
      far_buffer_.pop_front();
    }
    has_new_ = true;
  }

  // 定时器：拼接 near/far buffer → 处理
  void timerCallback()
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr current(new pcl::PointCloud<pcl::PointXYZ>);
    std_msgs::msg::Header header;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!has_new_) return;
      if (near_buffer_.empty() && far_buffer_.empty()) return;

      for (const auto & e : near_buffer_) *current += *e.cloud;
      for (const auto & e : far_buffer_)  *current += *e.cloud;

      if (!far_buffer_.empty())       header = far_buffer_.back().header;
      else if (!near_buffer_.empty()) header = near_buffer_.back().header;

      has_new_ = false;
    }

    if (current->empty()) return;

    const auto t0 = std::chrono::steady_clock::now();
    processCloud(current, header);
    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    RCLCPP_DEBUG(get_logger(), "processCloud took %ld ms, %zu pts", ms, current->size());
  }

  void processCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr & current,
                    const std_msgs::msg::Header & header)
  {
    if (!predefined_ || predefined_->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "predefined cloud not ready");
      return;
    }

    if (current->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "point empty");
      return;
    }

    // 1) 查询 TF：把先验点云从 predefined_frame_ 变到当前点云 frame
    const std::string target = header.frame_id;
    const std::string source = predefined_frame_;

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    if (!source.empty() && source != target) {
      try {
        auto tf_msg = tf_buffer_->lookupTransform(
          target, source,
          tf2::TimePointZero,
          tf2::durationFromSec(0.5));
        T = tf2::transformToEigen(tf_msg).matrix().cast<float>();
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "TF %s -> %s failed: %s",
                             source.c_str(), target.c_str(), ex.what());
        return;
      }
    }

    // 2) 变换先验点云 + 建 KD-Tree
    pcl::PointCloud<pcl::PointXYZ>::Ptr predefined_t(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*predefined_, *predefined_t, T);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(predefined_t);

    // 3) 最近邻 + 阈值剔除
    pcl::PointIndices::Ptr to_remove(new pcl::PointIndices);
    std::vector<int>   nn_idx(1);
    std::vector<float> nn_sq(1);
    const float th2 = static_cast<float>(threshold_ * threshold_);

    for (size_t i = 0; i < current->size(); ++i) {
      if (kdtree.nearestKSearch(current->points[i], 1, nn_idx, nn_sq) > 0 &&
          nn_sq[0] < th2) {
        to_remove->indices.push_back(static_cast<int>(i));
      }
    }

    // 4) 提取 remaining / removed
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(current);
    extract.setIndices(to_remove);

    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(true);
    extract.filter(*remaining);

    pcl::PointCloud<pcl::PointXYZ>::Ptr removed(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(false);
    extract.filter(*removed);

    // 5) 发布 removed
    {
      sensor_msgs::msg::PointCloud2 out;
      pcl::toROSMsg(*removed, out);
      out.header = header;
      pub_removed_->publish(out);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "current=%zu removed=%zu remaining=%zu",
                         current->size(), removed->size(), remaining->size());

    // 6) 聚类前降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining_ds(new pcl::PointCloud<pcl::PointXYZ>);
    {
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(remaining);
      vg.setLeafSize(0.02f, 0.02f, 0.02f);
      vg.filter(*remaining_ds);
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(remaining_ds);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.35);
    ec.setMinClusterSize(30);
    ec.setMaxClusterSize(30000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(remaining_ds);
    ec.extract(cluster_indices);

    std::vector<Eigen::Vector4f> centroids;
    centroids.reserve(cluster_indices.size());
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
      const auto & ci = cluster_indices[i];
      if (ci.indices.empty()) continue;
      Eigen::Vector4f c;
      pcl::compute3DCentroid(*remaining_ds, ci.indices, c);
      centroids.push_back(c);
    }

    auto confirmed = consistency_.update(centroids);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "cluster sum:%zu confirmed:%zu", centroids.size(), confirmed.size());

    float merge_dist = 0.15f;
    std::vector<Eigen::Vector4f> final_centroids = mergeCentroids(confirmed, merge_dist);

    // 7) 发布质心 PoseArray
    {
      geometry_msgs::msg::PoseArray pose_arr;
      pose_arr.header = header;
      pose_arr.poses.reserve(final_centroids.size());
      for (const auto & c : final_centroids) {
        geometry_msgs::msg::Pose p;
        p.position.x = c[0];
        p.position.y = c[1];
        p.position.z = c[2];
        p.orientation.w = 1.0;
        pose_arr.poses.push_back(p);
      }
      pub_centroids_->publish(pose_arr);
    }

    // 8) 发布 remaining 点云
    {
      sensor_msgs::msg::PointCloud2 remaining_msg;
      pcl::toROSMsg(*remaining, remaining_msg);
      remaining_msg.header = header;
      pub_->publish(remaining_msg);
    }

    // 9) MarkerArray
    {
      visualization_msgs::msg::MarkerArray marker_arr;

      for (size_t i = final_centroids.size(); i < last_marker_count_; ++i) {
        visualization_msgs::msg::Marker del;
        del.header = header;
        del.ns = "cluster_centroids";
        del.id = static_cast<int>(i);
        del.action = visualization_msgs::msg::Marker::DELETE;
        marker_arr.markers.push_back(del);
      }

      for (size_t i = 0; i < final_centroids.size(); ++i) {
        visualization_msgs::msg::Marker m;
        m.header = header;
        m.ns = "cluster_centroids";
        m.id = static_cast<int>(i);
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = final_centroids[i][0];
        m.pose.position.y = final_centroids[i][1];
        m.pose.position.z = final_centroids[i][2];
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.2;
        m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 0.0f; m.color.a = 1.0f;
        m.lifetime = rclcpp::Duration::from_seconds(1.0);
        marker_arr.markers.push_back(m);
      }
      last_marker_count_ = final_centroids.size();   // 用融合后的数量
      pub_centroids_marker_->publish(marker_arr);
    }
  }

  // 质心级融合：把因角分辨率断裂的同一目标的质心合并
  std::vector<Eigen::Vector4f> mergeCentroids(
      const std::vector<Eigen::Vector4f>& raw_centroids,
      float merge_distance)
  {
      std::vector<Eigen::Vector4f> merged;
      std::vector<bool> used(raw_centroids.size(), false);

      for (size_t i = 0; i < raw_centroids.size(); ++i) {
          if (used[i]) continue;
          Eigen::Vector4f sum = raw_centroids[i];
          int count = 1;
          used[i] = true;

          for (size_t j = i + 1; j < raw_centroids.size(); ++j) {
              if (used[j]) continue;
              float dist = (raw_centroids[i].head<3>() - raw_centroids[j].head<3>()).norm();
              if (dist < merge_distance) {
                  sum += raw_centroids[j];
                  count++;
                  used[j] = true;
              }
          }
          sum /= static_cast<float>(count);
          merged.push_back(sum);
      }
      return merged;
  }

  // ---------- 成员 ----------
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  FrameConsistencyFilter consistency_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr    pub_centroids_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_, pub_removed_;
  rclcpp::TimerBase::SharedPtr timer_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr predefined_;

  std::mutex mutex_;
  std::deque<CloudEntry> near_buffer_;
  std::deque<CloudEntry> far_buffer_;
  bool has_new_ = false;

  std::string predefined_path_, predefined_frame_;
  std::string input_topic_, output_topic_, removed_topic_;
  double threshold_         = 0.1;
  double process_rate_      = 10.0;
  double near_distance_     = 8.0;
  int    near_accum_frames_ = 2;
  int    far_accum_frames_  = 5;

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