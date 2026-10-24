// 真实规模实测：当前帧 ≈ 1 万点/帧 × 累积帧数
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b)
{ return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char ** argv)
{
  const int n_live = (argc > 1) ? atoi(argv[1]) : 70000;   // 7 帧 × 1 万
  const double yaw_deg = (argc > 2) ? atof(argv[2]) : 1.0;
  const double th = (argc > 3) ? atof(argv[3]) : 0.10;

  pcl::PointCloud<pcl::PointXYZ>::Ptr map(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile<pcl::PointXYZ>(
    "/home/tjurm/LIDAR_lidar/src/gicp/pcd/cloud_once.pcd", *map);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pre(new pcl::PointCloud<pcl::PointXYZ>);
  { pcl::VoxelGrid<pcl::PointXYZ> vg; vg.setInputCloud(map);
    vg.setLeafSize(0.02f,0.02f,0.02f); vg.filter(*pre); }

  // 当前帧：把地图转一个小角度当作"传感器动了"
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  const double a = yaw_deg * M_PI / 180.0;
  T(0,0)=cos(a); T(0,1)=-sin(a); T(1,0)=sin(a); T(1,1)=cos(a);
  pcl::PointCloud<pcl::PointXYZ>::Ptr all(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*map, *all, T);

  std::mt19937 rng(42);
  std::vector<size_t> order(all->size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::shuffle(order.begin(), order.end(), rng);
  pcl::PointCloud<pcl::PointXYZ>::Ptr live(new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < n_live && i < (int)all->size(); ++i) live->push_back(all->points[order[i]]);

  // 每周期成本
  auto t0 = clk::now();
  pcl::KdTreeFLANN<pcl::PointXYZ> kd;
  kd.setInputCloud(pre);
  auto t1 = clk::now();
  const double t_build = ms(t0, t1);

  std::vector<int> idx(1); std::vector<float> d2(1);
  const float th2 = (float)(th * th);
  pcl::PointCloud<pcl::PointXYZ>::Ptr rem(new pcl::PointCloud<pcl::PointXYZ>);
  t0 = clk::now();
  for (size_t i = 0; i < live->size(); ++i) {
    if (kd.nearestKSearch(live->points[i], 1, idx, d2) > 0 && d2[0] < th2) continue;
    rem->push_back(live->points[i]);
  }
  t1 = clk::now();
  const double t_nn = ms(t0, t1);

  pcl::PointCloud<pcl::PointXYZ>::Ptr rem_ds(new pcl::PointCloud<pcl::PointXYZ>);
  { pcl::VoxelGrid<pcl::PointXYZ> vg; vg.setInputCloud(rem);
    vg.setLeafSize(0.02f,0.02f,0.02f); vg.filter(*rem_ds); }

  t0 = clk::now();
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(rem_ds);
  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(0.35); ec.setMinClusterSize(30); ec.setMaxClusterSize(30000);
  ec.setSearchMethod(tree); ec.setInputCloud(rem_ds); ec.extract(clusters);
  t1 = clk::now();
  const double t_clu = ms(t0, t1);

  printf("地图 %zu -> %zu 点 | 当前帧 %zu 点 | 旋转 %.2f° | 阈值 %.2f m\n",
         map->size(), pre->size(), live->size(), yaw_deg, th);
  printf("  建 KD-tree      %7.1f ms\n", t_build);
  printf("  逐点 NN         %7.1f ms\n", t_nn);
  printf("  欧式聚类        %7.1f ms\n", t_clu);
  printf("  ---------- 合计 %7.1f ms  -> 上限 %.1f Hz\n",
         t_build + t_nn + t_clu, 1000.0 / (t_build + t_nn + t_clu));
  printf("  残余 %zu 点 (%.2f%%)  -> %zu 簇\n",
         rem->size(), 100.0 * rem->size() / live->size(), clusters.size());
  return 0;
}
