// 只读性能实测：复刻 pcl_process.cpp 每周期做的事，测量各阶段耗时
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <chrono>
#include <cstdio>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr map(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(
        "/home/tjurm/LIDAR_lidar/src/gicp/pcd/cloud_once.pcd", *map) != 0) {
    printf("load fail\n");
    return 1;
  }
  printf("地图载入: %zu 点\n", map->size());

  // A. 载入时 2cm 体素（构造函数里做的事）
  auto t0 = clk::now();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pre(new pcl::PointCloud<pcl::PointXYZ>);
  {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(map);
    vg.setLeafSize(0.02f, 0.02f, 0.02f);
    vg.filter(*pre);
  }
  auto t1 = clk::now();
  printf("A 载入时体素 0.02 : %zu 点   %.1f ms（只做一次）\n", pre->size(), ms(t0, t1));

  // B. 每周期：transformPointCloud（T=I 也照做）+ 建 KD-tree
  t0 = clk::now();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pre_t(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*pre, *pre_t, Eigen::Matrix4f::Identity());
  t1 = clk::now();
  printf("B1 每周期 transformPointCloud(I): %.1f ms\n", ms(t0, t1));

  t0 = clk::now();
  pcl::KdTreeFLANN<pcl::PointXYZ> kd;
  kd.setInputCloud(pre_t);
  t1 = clk::now();
  printf("B2 每周期 重建 KdTreeFLANN:        %.1f ms   <<< 关键\n", ms(t0, t1));

  // C. 构造一份"当前帧"：把地图转 0.5° 当输入，模拟真实数据
  Eigen::Matrix4f Tcur = Eigen::Matrix4f::Identity();
  const double a = 0.5 * M_PI / 180.0;
  Tcur(0, 0) = cos(a); Tcur(0, 1) = -sin(a);
  Tcur(1, 0) = sin(a); Tcur(1, 1) = cos(a);
  pcl::PointCloud<pcl::PointXYZ>::Ptr live(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*map, *live, Tcur);

  // D. 逐点 NN（threshold 0.1 → th2 = 0.01）
  std::vector<int> idx(1); std::vector<float> d2(1);
  t0 = clk::now();
  size_t removed = 0;
  for (size_t i = 0; i < live->size(); ++i) {
    if (kd.nearestKSearch(live->points[i], 1, idx, d2) > 0 && d2[0] < 0.01f) removed++;
  }
  t1 = clk::now();
  printf("D  %zu 点逐点 NN (th=0.10):       %.1f ms   残余 %zu (%.1f%%)\n",
         live->size(), ms(t0, t1), live->size() - removed,
         100.0 * (live->size() - removed) / live->size());

  // E. 残余点 → 2cm 体素 → 欧式聚类（tol 0.35 / min 30）
  pcl::PointCloud<pcl::PointXYZ>::Ptr rem(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < live->size(); ++i) {
    if (kd.nearestKSearch(live->points[i], 1, idx, d2) > 0 && d2[0] >= 0.01f)
      rem->push_back(live->points[i]);
  }
  t0 = clk::now();
  pcl::PointCloud<pcl::PointXYZ>::Ptr rem_ds(new pcl::PointCloud<pcl::PointXYZ>);
  {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(rem);
    vg.setLeafSize(0.02f, 0.02f, 0.02f);
    vg.filter(*rem_ds);
  }
  t1 = clk::now();
  printf("E1 残余点 2cm 体素: %zu -> %zu     %.1f ms\n", rem->size(), rem_ds->size(), ms(t0, t1));

  t0 = clk::now();
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(rem_ds);
  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(0.35);
  ec.setMinClusterSize(30);
  ec.setMaxClusterSize(30000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(rem_ds);
  ec.extract(clusters);
  t1 = clk::now();
  printf("E2 欧式聚类(tol0.35/min30):       %.1f ms   -> %zu 簇\n",
         ms(t0, t1), clusters.size());
  return 0;
}
