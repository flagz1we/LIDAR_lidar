// 来源: https://github.com/JaminJiang/dbscan_kdtree
// 轻量级 DBSCAN 实现，使用 KD-Tree 加速邻域搜索
#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <cmath>

namespace dbscan {

template <typename PointT>
class DBSCAN {
public:
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;

    DBSCAN(double eps, int min_pts) : eps_(eps), min_pts_(min_pts) {}

    std::vector<std::vector<int>> run(const PointCloudPtr& cloud) {
        tree_.setInputCloud(cloud);
        cloud_ = cloud;
        std::vector<int> labels(cloud->size(), -1); // -1 = 未分类
        int cluster_id = 0;

        for (size_t i = 0; i < cloud->size(); ++i) {
            if (labels[i] != -1) continue;
            std::vector<int> neighbors;
            radiusSearch(i, neighbors);
            if (static_cast<int>(neighbors.size()) < min_pts_) {
                labels[i] = -2; // 噪声
            } else {
                expandCluster(i, neighbors, cluster_id, labels);
                cluster_id++;
            }
        }

        // 收集每个簇的点索引
        std::vector<std::vector<int>> clusters(cluster_id);
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] >= 0) clusters[labels[i]].push_back(i);
        }
        return clusters;
    }

private:
    void expandCluster(int core_idx, std::vector<int>& seeds, int cluster_id, std::vector<int>& labels) {
        labels[core_idx] = cluster_id;
        for (size_t i = 0; i < seeds.size(); ++i) {
            int idx = seeds[i];
            if (labels[idx] == -2) labels[idx] = cluster_id; // 噪声点变为边界点
            if (labels[idx] != -1) continue; // 已分类
            labels[idx] = cluster_id;
            std::vector<int> new_neighbors;
            radiusSearch(idx, new_neighbors);
            if (static_cast<int>(new_neighbors.size()) >= min_pts_) {
                seeds.insert(seeds.end(), new_neighbors.begin(), new_neighbors.end());
            }
        }
    }

    void radiusSearch(int idx, std::vector<int>& neighbors) {
        std::vector<int> idx_n;
        std::vector<float> dist_n;
        tree_.radiusSearch(cloud_->points[idx], static_cast<float>(eps_), idx_n, dist_n);
        neighbors.insert(neighbors.end(), idx_n.begin(), idx_n.end());
    }

    double eps_;
    int min_pts_;
    PointCloudPtr cloud_;
    pcl::KdTreeFLANN<PointT> tree_;
};

} // namespace dbscan