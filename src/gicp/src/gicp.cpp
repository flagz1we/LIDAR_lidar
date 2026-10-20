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
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <mutex>

class GICPLocalization : public rclcpp::Node {
public:
    GICPLocalization() : Node("gicp_localization_node") {
        // ---------- 参数声明 ----------
        this->declare_parameter<std::string>("map_pcd", "");
        this->declare_parameter<double>("voxel_size", 0.2);
        this->declare_parameter<double>("max_corr_dist", 1.0);
        this->declare_parameter<int>("max_iter", 100);
        this->declare_parameter<double>("trans_eps", 1e-8);
        this->declare_parameter<double>("fit_eps", 1.0);
        this->declare_parameter<double>("crop_radius", 30.0);
        this->declare_parameter<std::string>("map_frame", "map");
        this->declare_parameter<std::string>("base_frame", "odom");
        this->declare_parameter<std::string>("odom_frame", "odom");
        this->declare_parameter<double>("registration_freq", 0.1);   // 配准频率

        // ---------- 获取参数 ----------
        std::string map_pcd = this->get_parameter("map_pcd").as_string();
        voxel_size_      = this->get_parameter("voxel_size").as_double();
        max_corr_dist_   = this->get_parameter("max_corr_dist").as_double();
        max_iter_        = this->get_parameter("max_iter").as_int();
        trans_eps_       = this->get_parameter("trans_eps").as_double();
        fit_eps_         = this->get_parameter("fit_eps").as_double();
        crop_radius_     = this->get_parameter("crop_radius").as_double();
        map_frame_       = this->get_parameter("map_frame").as_string();
        base_frame_      = this->get_parameter("base_frame").as_string();
        odom_frame_      = this->get_parameter("odom_frame").as_string();
        double reg_freq  = this->get_parameter("registration_freq").as_double();

        use_sim_time_ = this->get_parameter("use_sim_time").as_bool();
        if (use_sim_time_) {
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        }

        // ---------- 加载全局地图并降采样 ----------
        global_map_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_pcd, *global_map_) == -1) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file: %s", map_pcd.c_str());
        rclcpp::shutdown();
        return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded map: %zu points", global_map_->size());

        pcl::VoxelGrid<pcl::PointXYZ> voxel_map;
        voxel_map.setInputCloud(global_map_);
        voxel_map.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
        voxel_map.filter(*global_map_);
        RCLCPP_INFO(this->get_logger(), "Map downsampled: %zu points", global_map_->size());

        // ---------- 初始化 GICP ----------
        gicp_ = std::make_shared<pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>();
        gicp_->setMaxCorrespondenceDistance(max_corr_dist_);
        gicp_->setMaximumIterations(max_iter_);
        gicp_->setTransformationEpsilon(trans_eps_);
        gicp_->setEuclideanFitnessEpsilon(fit_eps_);

        // ---------- TF ----------
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // ---------- 订阅与发布 ----------
        // 点云回调只存储最新一帧，不做复杂处理
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/cloud_registered", rclcpp::SensorDataQoS(),
        std::bind(&GICPLocalization::cloudCallback, this, std::placeholders::_1));

        pub_aligned_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_cloud", 10);
        pub_pose_    = this->create_publisher<geometry_msgs::msg::PoseStamped>("/robot_pose", 10);

        // ---------- 定时器：按固定频率执行配准 ----------
        auto period = std::chrono::duration<double>(1.0 / reg_freq);
        timer_ = this->create_wall_timer(period, std::bind(&GICPLocalization::timerCallback, this));

        // 位姿记忆初始化
        T_map_odom_ = Eigen::Affine3d::Identity();
        last_pose_valid_ = false;

        RCLCPP_INFO(this->get_logger(), "GICP Localization Node started (timer at %.1f Hz)", reg_freq);
    }

private:
    // ========== 点云回调：仅存储最新点云（带互斥锁） ==========
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        latest_cloud_ = msg;   // 直接存储原始消息，避免不必要的拷贝
    }

    // ========== 定时器回调：执行完整的配准流程 ==========
    void timerCallback() {
        // 1. 取出最新点云（若空则跳过）
        sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            if (!latest_cloud_) {
                return;
            }
            cloud_msg = latest_cloud_;
        }

        // 2. 转换到 PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud_in);
        std::string lidar_frame = cloud_msg->header.frame_id;

        // 3. 获取初始猜测：直接从 TF 查 odom -> lidar
        Eigen::Affine3d T_odom_lidar;
        try {
            geometry_msgs::msg::TransformStamped tf_odom_lidar =
                tf_buffer_->lookupTransform(odom_frame_, lidar_frame, tf2::TimePointZero);
            T_odom_lidar = tf2::transformToEigen(tf_odom_lidar);
        } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Cannot get %s -> %s: %s. Skip this cycle.",
                    odom_frame_.c_str(), lidar_frame.c_str(), ex.what());
        return;
        }

        // 用记忆的 map->odom 得到 map->lidar 粗猜测
        Eigen::Affine3d T_map_lidar_guess = T_map_odom_ * T_odom_lidar;

        // 4. 将当前扫描变换到 map 坐标系（粗位置）
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_in_map(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud_in, *source_in_map, T_map_lidar_guess.matrix().cast<float>());

        // 5. 降采样
        pcl::VoxelGrid<pcl::PointXYZ> voxel_src;
        voxel_src.setInputCloud(source_in_map);
        voxel_src.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_src.filter(*source_filtered);
        if (source_filtered->empty()) {
        RCLCPP_WARN(this->get_logger(), "Source cloud empty after downsampling");
        return;
        }

        // 6. 裁剪局部地图（基于当前粗猜测位置）
        Eigen::Vector3f center = T_map_odom_.translation().cast<float>();
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(global_map_);
        crop.setMin(Eigen::Vector4f(center.x() - crop_radius_,
                                    center.y() - crop_radius_,
                                    center.z() - crop_radius_, 1.0));
        crop.setMax(Eigen::Vector4f(center.x() + crop_radius_,
                                    center.y() + crop_radius_,
                                    center.z() + crop_radius_, 1.0));
        pcl::PointCloud<pcl::PointXYZ>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZ>);
        crop.filter(*local_map);
        if (local_map->empty()) {
        RCLCPP_WARN(this->get_logger(), "Local map empty, check crop_radius");
        return;
        }

        // 7. GICP 配准 (初始修正为单位矩阵)
        gicp_->setInputSource(source_filtered);
        gicp_->setInputTarget(global_map_);
        pcl::PointCloud<pcl::PointXYZ> aligned;
        gicp_->align(aligned, Eigen::Matrix4f::Identity());
        

        // 8. 更新 map->odom 变换
        if (gicp_->hasConverged()) {
            
            Eigen::Affine3f T_correction(gicp_->getFinalTransformation());

            double trans_jump = T_correction.translation().norm();          // 平移量 (米)
            Eigen::AngleAxisf rot_aa(T_correction.rotation());              // 轴角表示
            double angle_jump = std::abs(rot_aa.angle());                   // 旋转角 (弧度)

            // 阈值：可根据实际环境调整
            const double max_trans_jump = 0.5;   // 最大平移跳变 (米)
            const double max_angle_jump = 0.2;   // 最大旋转跳变 (弧度，约 11.5°)

            if (trans_jump > max_trans_jump || angle_jump > max_angle_jump) {
                RCLCPP_WARN(this->get_logger(),
                            "Rejecting large jump: trans=%.3f m, angle=%.3f rad. Keeping previous pose.",
                            trans_jump, angle_jump);
            // 不更新 T_map_odom_，直接跳到发布部分
            }
            else{   
                // 更新 map->odom = T_correction * T_map_odom_ (注意乘法顺序: 修正发生在 map 系下)
                T_map_odom_ = Eigen::Affine3d((T_correction.matrix().cast<double>() * T_map_odom_.matrix()).eval());
                last_pose_valid_ = true;
                RCLCPP_INFO(this->get_logger(), "GICP converged, score: %f", gicp_->getFitnessScore());
            }

        } else {
        RCLCPP_WARN(this->get_logger(), "GICP did NOT converge, score: %f. Keep previous pose.",
                    gicp_->getFitnessScore());
        // T_map_odom_ 保持不变，继续使用上一次的估计
        }

        // 9. 发布 map -> odom 变换
        geometry_msgs::msg::TransformStamped tf_map_odom;
        tf_map_odom.header.stamp = this->now();
        tf_map_odom.header.frame_id = map_frame_;
        tf_map_odom.child_frame_id = odom_frame_;
        tf_map_odom.transform = tf2::eigenToTransform(T_map_odom_).transform;
        tf_broadcaster_->sendTransform(tf_map_odom);

        // 10. 发布机器人位姿 (map -> base_link)，方便查看
        try {
        geometry_msgs::msg::TransformStamped tf_odom_base =
            tf_buffer_->lookupTransform(odom_frame_, base_frame_, tf2::TimePointZero);
        Eigen::Affine3d T_odom_base = tf2::transformToEigen(tf_odom_base);
        Eigen::Affine3d T_map_base = T_map_odom_ * T_odom_base;
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = tf_map_odom.header.stamp;
        pose_msg.header.frame_id = map_frame_;
        pose_msg.pose.position.x = T_map_base.translation().x();
        pose_msg.pose.position.y = T_map_base.translation().y();
        pose_msg.pose.position.z = T_map_base.translation().z();
        Eigen::Quaterniond q(T_map_base.rotation());
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();
        pub_pose_->publish(pose_msg);
        } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "Cannot compute map->base_link: %s", ex.what());
        }

        // 11. 可选：发布对齐后的点云
        if (pub_aligned_->get_subscription_count() > 0) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (gicp_->hasConverged()) {
            pcl::transformPointCloud(*source_filtered, *final_cloud,
                                    gicp_->getFinalTransformation());
        } else {
            *final_cloud = *source_filtered;
        }
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*final_cloud, out_msg);
        out_msg.header.frame_id = map_frame_;
        out_msg.header.stamp = this->now();
        pub_aligned_->publish(out_msg);
        }
    }

    // ---------- 参数 ----------
    double voxel_size_, max_corr_dist_, trans_eps_, fit_eps_, crop_radius_;
    int max_iter_;
    std::string map_frame_, base_frame_, odom_frame_;
    bool use_sim_time_;

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
    Eigen::Affine3d T_map_odom_;       // 当前估计的 map->odom 变换
    bool last_pose_valid_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GICPLocalization>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}