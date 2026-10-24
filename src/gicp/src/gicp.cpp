#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>        // pcl::removeNaNFromPointCloud
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>
#include <mutex>

// =====================================================================================
// 坐标系约定（重要）
//
// 地图 cloud_once.pcd 是 save_pcd.py 从 /livox/lidar 采 5 s 累积得到的，
// 因此它表达在「录制时刻的 livox_frame」下，不是世界系全局地图。
// 本节点被跟踪的状态量统一叫 T_map_tracked_：
//   use_odom = false（默认，本机无任何里程计）
//       tracked = 当前点云的 header.frame_id，也就是 livox_frame
//       初值猜测 = 上一周期的 T_map_tracked_（没有运动预测可用）
//       发布 TF: map -> livox_frame
//   use_odom = true（有 FAST-LIO / Point-LIO 等给出 odom）
//       tracked = odom_frame
//       初值猜测 = T_map_tracked_ * T_odom_lidar
//       发布 TF: map -> odom_frame
// =====================================================================================

class GICPLocalization : public rclcpp::Node {
public:
    GICPLocalization() : Node("gicp_localization_node") {
        // ---------- 参数声明 ----------
        declare_parameter<std::string>("map_pcd", "/home/tjurm/LIDAR_lidar/src/gicp/pcd/cloud_once.pcd");
        declare_parameter<std::string>("input_topic", "/livox/lidar");
        declare_parameter<double>("voxel_size", 0.2);
        declare_parameter<double>("max_corr_dist", 1.0);
        declare_parameter<int>("max_iter", 60);
        declare_parameter<double>("trans_eps", 1e-6);   // 1e-8 对 float32 变换矩阵不可达，只会永远跑满迭代
        declare_parameter<double>("crop_radius", 30.0);
        declare_parameter<std::string>("map_frame", "map");
        declare_parameter<bool>("use_odom", false);     // 本机没有 odom 坐标系，默认 false
        declare_parameter<std::string>("odom_frame", "odom");
        declare_parameter<std::string>("base_frame", "");  // 留空 = 直接发雷达自身位姿（本机无 base_link）
        declare_parameter<double>("registration_freq", 1.0);
        declare_parameter<double>("max_fitness_score", 0.05);  // 适应度阈值，单位 m^2（均方差距离）
        declare_parameter<double>("max_trans_jump", 0.5);      // 单周期修正平移上限 (m)
        declare_parameter<double>("max_angle_jump", 0.2);      // 单周期修正旋转上限 (rad, 约 11.5°)
        declare_parameter<int>("min_input_points", 100);       // 少于这个数就别配了

        // ---------- 获取参数 ----------
        const std::string map_pcd = get_parameter("map_pcd").as_string();
        input_topic_       = get_parameter("input_topic").as_string();
        voxel_size_        = get_parameter("voxel_size").as_double();
        max_corr_dist_     = get_parameter("max_corr_dist").as_double();
        max_iter_          = get_parameter("max_iter").as_int();
        trans_eps_         = get_parameter("trans_eps").as_double();
        crop_radius_       = get_parameter("crop_radius").as_double();
        map_frame_         = get_parameter("map_frame").as_string();
        use_odom_          = get_parameter("use_odom").as_bool();
        odom_frame_        = get_parameter("odom_frame").as_string();
        base_frame_        = get_parameter("base_frame").as_string();
        max_fitness_score_ = get_parameter("max_fitness_score").as_double();
        max_trans_jump_    = get_parameter("max_trans_jump").as_double();
        max_angle_jump_    = get_parameter("max_angle_jump").as_double();
        min_input_points_  = get_parameter("min_input_points").as_int();
        double reg_freq    = get_parameter("registration_freq").as_double();
        if (reg_freq <= 0.0) reg_freq = 1.0;

        // ---------- 加载全局地图并降采样 ----------
        global_map_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_pcd, *global_map_) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file: %s", map_pcd.c_str());
            rclcpp::shutdown();
            return;
        }
        // NaN/Inf 会毒化 GICP 的协方差估计，必须先剔除
        {
            std::vector<int> valid_idx;
            pcl::removeNaNFromPointCloud(*global_map_, *global_map_, valid_idx);
        }
        RCLCPP_INFO(this->get_logger(), "Loaded map: %zu points", global_map_->size());

        pcl::VoxelGrid<pcl::PointXYZ> voxel_map;
        voxel_map.setInputCloud(global_map_);
        voxel_map.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
        voxel_map.filter(*global_map_);
        RCLCPP_INFO(this->get_logger(), "Map downsampled: %zu points", global_map_->size());
        if (global_map_->size() < 50) {
            RCLCPP_ERROR(this->get_logger(), "Map has too few points (%zu), abort.", global_map_->size());
            rclcpp::shutdown();
            return;
        }

        // ---------- 初始化 GICP ----------
        gicp_ = std::make_shared<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
        gicp_->setMaxCorrespondenceDistance(max_corr_dist_);
        gicp_->setMaximumIterations(max_iter_);
        gicp_->setTransformationEpsilon(trans_eps_);
        // 注意：GICP 不使用 setEuclideanFitnessEpsilon（见 PCL impl/gicp.hpp），故不再设置

        // ---------- TF ----------
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // ---------- 订阅与发布 ----------
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, rclcpp::SensorDataQoS(),
            std::bind(&GICPLocalization::cloudCallback, this, std::placeholders::_1));

        // 绝不能发到 /livox/lidar：那是雷达驱动的话题，混进 map 系的云会把下游（LIO）喂坏
        pub_aligned_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/gicp/aligned_cloud", 10);
        pub_pose_    = this->create_publisher<geometry_msgs::msg::PoseStamped>("/robot_pose", 10);

        // ---------- 定时器 ----------
        auto period = std::chrono::duration<double>(1.0 / reg_freq);
        timer_ = this->create_wall_timer(period, std::bind(&GICPLocalization::timerCallback, this));

        // ---------- 状态初始化 ----------
        T_map_tracked_ = Eigen::Affine3d::Identity();

        RCLCPP_INFO(this->get_logger(),
                    "GICP Localization started: topic=%s, use_odom=%s, timer=%.2f Hz, "
                    "map=%zu pts, fitness_gate=%.4f m^2",
                    input_topic_.c_str(), use_odom_ ? "true" : "false", reg_freq,
                    global_map_->size(), max_fitness_score_);
        if (!use_odom_) {
            RCLCPP_INFO(this->get_logger(),
                        "use_odom=false: 初值只用上一周期位姿，发布 %s -> <点云 frame_id>。"
                        "要求机器人起始位置接近地图录制位置。", map_frame_.c_str());
        }
    }

private:
    // ========== 点云回调：只缓存最新一帧 ==========
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        latest_cloud_ = msg;
    }

    // ========== 定时器回调：完整配准流程 ==========
    void timerCallback() {
        // ---------- 1. 取最新点云 ----------
        sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            if (!latest_cloud_) {
                // 原版这里是静默 return，导致「没数据」和「配准失败」长得一模一样
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Waiting for point cloud on '%s' (matched publishers: %zu). No data -> no registration.",
                    input_topic_.c_str(), sub_cloud_->get_publisher_count());
                return;
            }
            cloud_msg = latest_cloud_;
        }

        // ---------- 2. 转 PCL，剔 NaN ----------
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud_in);
        {
            std::vector<int> valid_idx;
            pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, valid_idx);
        }
        if (static_cast<int>(cloud_in->size()) < min_input_points_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Input cloud too small after NaN removal (%zu < %d), skip.",
                cloud_in->size(), min_input_points_);
            return;
        }

        // ---------- 3. 点云所在坐标系 ----------
        std::string lidar_frame = cloud_msg->header.frame_id;
        if (lidar_frame.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Cloud header.frame_id is empty, skip this cycle.");
            return;
        }

        // ---------- 4. 初值猜测 ----------
        Eigen::Affine3d T_odom_lidar = Eigen::Affine3d::Identity();
        std::string tracked_frame = lidar_frame;   // 要广播的 child frame
        if (use_odom_) {
            try {
                geometry_msgs::msg::TransformStamped tf_odom_lidar =
                    tf_buffer_->lookupTransform(odom_frame_, lidar_frame, tf2::TimePointZero);
                T_odom_lidar = tf2::transformToEigen(tf_odom_lidar);
                tracked_frame = odom_frame_;
            } catch (tf2::TransformException & ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Cannot get %s -> %s: %s. Skip this cycle.",
                    odom_frame_.c_str(), lidar_frame.c_str(), ex.what());
                return;
            }
        }
        const Eigen::Affine3d T_map_lidar_guess = T_map_tracked_ * T_odom_lidar;

        // ---------- 5. 变换到 map 系 + 降采样 ----------
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_in_map(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud_in, *source_in_map, T_map_lidar_guess.matrix().cast<float>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        {
            pcl::VoxelGrid<pcl::PointXYZ> voxel_src;
            voxel_src.setInputCloud(source_in_map);
            voxel_src.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
            voxel_src.filter(*source_filtered);
        }
        if (source_filtered->size() < 50) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Source cloud too small after downsampling (%zu), skip.", source_filtered->size());
            return;
        }

        // ---------- 6. 裁剪局部地图 ----------
        // 中心必须用「本帧雷达在 map 系下的位置」，原版用 T_map_odom_ 的平移（odom 原点）是错的
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        {
            const Eigen::Vector3f c = T_map_lidar_guess.translation().cast<float>();
            pcl::CropBox<pcl::PointXYZ> crop;
            crop.setInputCloud(global_map_);
            crop.setMin(Eigen::Vector4f(c.x() - static_cast<float>(crop_radius_),
                                        c.y() - static_cast<float>(crop_radius_),
                                        c.z() - static_cast<float>(crop_radius_), 1.0f));
            crop.setMax(Eigen::Vector4f(c.x() + static_cast<float>(crop_radius_),
                                        c.y() + static_cast<float>(crop_radius_),
                                        c.z() + static_cast<float>(crop_radius_), 1.0f));
            crop.filter(*target_cloud);
        }
        if (target_cloud->size() < 50) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Local map too small (%zu) around guess, fallback to global map.", target_cloud->size());
            target_cloud = global_map_;
        }

        // ---------- 7. GICP 配准（初始修正为单位阵） ----------
        gicp_->setInputSource(source_filtered);
        gicp_->setInputTarget(target_cloud);
        pcl::PointCloud<pcl::PointXYZ> aligned;
        gicp_->align(aligned, Eigen::Matrix4f::Identity());

        // ---------- 8. 质量判据 ----------
        // 注意：PCL 1.12 的 GICP 在 nr_iterations_ >= max_iterations_ 时就把 converged_ 置 true
        //      (impl/gicp.hpp:496)，所以 hasConverged() 基本恒为 true，不能当质量判据。
        //      真正的判据是 fitness score（均方差距离）与修正量大小。
        const Eigen::Matrix4f T_corr_f = gicp_->getFinalTransformation();
        const double fitness = gicp_->getFitnessScore(max_corr_dist_);

        bool accept = false;
        std::string reject_reason;

        if (!T_corr_f.allFinite() || !std::isfinite(fitness)) {
            reject_reason = "non-finite result";
        } else {
            const Eigen::Affine3f T_corr(T_corr_f);
            const double trans_jump = T_corr.translation().norm();
            const Eigen::AngleAxisf rot_aa(T_corr.rotation());
            const double angle_jump = std::abs(rot_aa.angle());

            if (fitness > max_fitness_score_) {
                reject_reason = "fitness too high (" + std::to_string(fitness) + " m^2)";
            } else if (trans_jump > max_trans_jump_) {
                reject_reason = "translation jump too large";
            } else if (angle_jump > max_angle_jump_) {
                reject_reason = "rotation jump too large";
            } else {
                accept = true;
                T_map_tracked_ = Eigen::Affine3d(
                    (T_corr.matrix().cast<double>() * T_map_tracked_.matrix()).eval());
                last_score_ = fitness;

                RCLCPP_INFO(this->get_logger(),
                    "GICP ok: score=%.4f m^2 (rms=%.3f m), corr: t=%.3f m / %.2f deg, "
                    "pose: x=%.3f y=%.3f z=%.3f",
                    fitness, std::sqrt(fitness), trans_jump,
                    angle_jump * 180.0 / M_PI,
                    T_map_tracked_.translation().x(),
                    T_map_tracked_.translation().y(),
                    T_map_tracked_.translation().z());
            }
        }

        if (!accept) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Reject this cycle: %s (score=%.4f, last good=%.4f). Keep previous pose.",
                reject_reason.c_str(), fitness, last_score_);
        }

        // ---------- 9. 发布 map -> tracked_frame ----------
        const rclcpp::Time stamp = this->now();
        geometry_msgs::msg::TransformStamped tf_map_tracked;
        tf_map_tracked.header.stamp = stamp;
        tf_map_tracked.header.frame_id = map_frame_;
        tf_map_tracked.child_frame_id = tracked_frame;
        tf_map_tracked.transform = tf2::eigenToTransform(T_map_tracked_).transform;
        tf_broadcaster_->sendTransform(tf_map_tracked);

        // ---------- 10. 发布位姿 ----------
        // base_frame 为空（本机没有 base_link）或查不到时，退化为「雷达在 map 中的位姿」
        Eigen::Affine3d T_map_base = T_map_tracked_;
        if (!base_frame_.empty()) {
            try {
                const geometry_msgs::msg::TransformStamped tf_tracked_base =
                    tf_buffer_->lookupTransform(tracked_frame, base_frame_, tf2::TimePointZero);
                T_map_base = T_map_tracked_ * tf2::transformToEigen(tf_tracked_base);
            } catch (tf2::TransformException & ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Cannot get %s -> %s (%s), publish sensor pose instead.",
                    tracked_frame.c_str(), base_frame_.c_str(), ex.what());
            }
        }

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = map_frame_;
        pose_msg.pose.position.x = T_map_base.translation().x();
        pose_msg.pose.position.y = T_map_base.translation().y();
        pose_msg.pose.position.z = T_map_base.translation().z();
        const Eigen::Quaterniond q(T_map_base.rotation());
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();
        pub_pose_->publish(pose_msg);

        // ---------- 11. 发布对齐后的点云（map 系） ----------
        if (pub_aligned_->get_subscription_count() > 0) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*source_filtered, *final_cloud, T_corr_f);

            sensor_msgs::msg::PointCloud2 out_msg;
            pcl::toROSMsg(*final_cloud, out_msg);
            out_msg.header.frame_id = map_frame_;
            out_msg.header.stamp = stamp;
            pub_aligned_->publish(out_msg);
        }
    }

    // ---------- 参数 ----------
    std::string input_topic_, map_frame_, odom_frame_, base_frame_;
    double voxel_size_ = 0.04, max_corr_dist_ = 1.0, trans_eps_ = 1e-6, crop_radius_ = 30.0;
    double max_fitness_score_ = 0.05, max_trans_jump_ = 0.5, max_angle_jump_ = 0.2;
    int max_iter_ = 60;
    int min_input_points_ = 100;
    bool use_odom_ = false;

    // ---------- 地图 ----------
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_;

    // ---------- GICP ----------
    std::shared_ptr<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>> gicp_;

    // ---------- TF ----------
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ---------- 订阅与发布 ----------
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_aligned_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ---------- 数据缓存 ----------
    sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_;
    std::mutex cloud_mutex_;

    // ---------- 定位状态 ----------
    Eigen::Affine3d T_map_tracked_;   // map -> (odom 或 点云 frame)
    double last_score_ = -1.0;        // 最近一次被接受的 fitness
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GICPLocalization>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
