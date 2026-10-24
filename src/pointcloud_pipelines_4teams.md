# 四个队伍的点云处理链路（逐行阅读源码整理）

阅读范围与文件规模：

| 项目 | 语言/框架 | 核心文件 | 行数 |
|---|---|---|---|
| T-DT_Radar（东北大学） | C++ / ROS 2 | `src/lidar/{localization,dynamic_cloud,cluster}/src/*.cpp` | 187 + 317 + 84 |
| EGA_Radar_Algorithm | C++ / ROS 2 | `Lidar_ws/src/{localization,dynamic_cloud,cluster}/src/*.cpp` | 1554 + 757 + 283 |
| ROBOMASTER-HORIZON-LiDAR-2026 | C++ / ROS 1 | `2026_pclmatcher_ws/src/pclmatcher/src/pclmatcher.cpp` | 948 |
| TJURM-Radar（radar-base） | C++ / ROS 1 | `src/main.cpp` + `src/fuse/fuse.cpp` | 377 + 410 |

> 一个重要发现：**EGA 的 lidar 栈是 fork 自 T-DT 的**——`EGA_Radar_Algorithm-main/Lidar_ws/src/cluster/src/cluster.cpp:283` 最后一行仍然是
> `RCLCPP_COMPONENTS_REGISTER_NODE(tdt_radar::Cluster)`，命名空间没改。这解释了为什么两边聚类容差都是 0.25 m。

---

## 一、T-DT_Radar —— 三节点串联，最完整的一条

### 节点 1：`localization` —— 一次性配准，之后位姿固定

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | `localization.cpp:16-19` | 载入场地 PCD `config/RM2025.pcd` | — |
| 2 | `:58-60` | 订阅 `/livox/lidar` | — |
| 3 | `:57` | `if (!has_aligned_)` —— **只跑一次**的门闩 | — |
| 4 | `:62-68` | 累积点云，攒够才继续 | `accumulate_time = 20` (`:176`) |
| 5 | `:76-77` | 转球面角：`azimuth=atan2(y,x)`、`elevation=atan2(z,√(x²+y²))` | `gridSizeDegrees = 0.1°` (`:181`) |
| 6 | `:83-94` | **每格只保留距离最远的点**（`Grid{farthestPoint, maxDistance}`） | — |
| 7 | `:100-107` | ROI 裁剪 | x∈(5,30)，y∈(-10,8)，z<7 |
| 8 | `:109-117` | 地图与源点云都体素下采样 | 0.1 m |
| 9 | `:120-121` | GICP 配准 | 默认参数 |
| 10 | `:142-144` | `fitness < 0.2` 才置 `has_aligned_ = true` | 0.2 |
| 11 | `:163-183` | 每帧广播静态 TF `rm_frame → livox_frame` | — |

**关键设计**：第 6 步「每格取最远点」是**抗动态目标的配准去噪**——动态目标挡在背景前面，同一角度格子里最远的回波才是静止结构。拿这个去跑 GICP 不会被移动物体带偏。

### 节点 2：`dynamic_cloud` —— 静态地图 KNN 差分

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | `dynamic_cloud.cpp:16-27` | 载入地图 → 体素 → 建 KD-tree（**只建一次**） | 0.1 m |
| 2 | `:166-167` | `lookupTransform("rm_frame", msg->header.frame_id, tf2::TimePointZero)` | — |
| 3 | `:175-184` | 变换到 `rm_frame` | — |
| 4 | `:187-207` | ROI + 禁区剔除 | x∈[3,28]，y∈[0,15]，**z∈[0,1.4]** |
| 5 | `:208-209` | `GetDynamicCloud(filtered, out, 0.1, 12)` | 阈值 0.1、12 线程 |
| 6 | `:46` / `:58-63` | KNN `K=1`，`pointNKNSquaredDistance[0] > threshold` 判动态 | **阈值是平方距离** |
| 7 | `:52-67` | 12 个 `std::thread` 分段并行 KNN | — |
| 8 | `:211-222` | 累积滑窗 | `accumulate_time = 3`（头文件） |
| 9 | `:234-243` | 发布 `/livox/lidar_dynamic`，frame=`rm_frame`，**stamp 沿用输入** | — |
| 10 | `:135-156` / `:244-306` | 飞镖/无人机空间盒子判定（z∈[1.7,3] 独立处理） | 点数 >5 / >40 触发 |

**注意第 6 步的单位陷阱**：`nearestKSearch` 返回的是**平方距离**，传 `threshold=0.1` 意味着实际距离
$$\sqrt{0.1}=0.316\ \text{m}$$
不是 0.1 m。这个「有效阈值 0.316 m」是全套链路里最宽的一环。

### 节点 3：`cluster` —— 欧式聚类取质心

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | `cluster.cpp:20-23` | 订阅 `/livox/lidar_dynamic` | — |
| 2 | `:43-53` | `EuclideanClusterExtraction` | **tol 0.25**，min 5，max 1000 |
| 3 | `:56-76` | 质心 = 逐坐标算术平均 | — |
| 4 | `:77-82` | 发布 `/livox/lidar_cluster`（质心点云） | — |

---

## 二、EGA_Radar_Algorithm —— 与 T-DT 同源，但加了分区

### 节点 1：`localization`（1554 行）

- 配准链：`backend: gicp_ndt` = GICP → 粗 NDT(0.5 m) → 细 NDT(0.2 m)
- 另有 0.1° 角度栅格背景：`background_range_tolerance: 0.10`、`background_candidate_tolerance: 0.15`
- 该栅格**只用于配准 source**，结果发到 `/filter_map`，全仓无人订阅（纯调试）

### 节点 2：`dynamic_cloud`（757 行）

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | 构造期 | 载入 `RMUC2026.pcd`（358,525 点）→ 体素 → 建树（一次） | 0.1 m |
| 2 | `:567-577` | `lookupTransform("rm_frame", ≤frame_id>, TimePointZero)` + 变换 | — |
| 3 | `:586-607` | **先删手写禁止区** | `dart_exclude_roi`、`home_exclude_roi`、`map_exclude_rotated_roi`、4 组 `map_exclude_axis_aligned_rois` |
| 4 | `:607-620` | `main_roi` 外的点分流到「空中层」`other/fly` | `main_roi: [0.3, 27.4, 0.3, 14.7, 0.1, 1.3]` |
| 5 | `:621-623` | `GetDynamicCloud(filtered, out, 12)` | 12 线程 |
| 6 | `:438-480` | 并行 KNN `K=1`，**逐点按所在区取阈值** | 见下 |
| 7 | `:382-403` | `GetDynamicRegion`：按 x 分区 | `split_axis="x"`，16.0 / 22.0，`near_region_positive=false` |
| 8 | `:405-418` | 分区阈值 | near 0.05 / middle 0.04 / far 0.03（**m²**） |
| 9 | `:625-650` | 按区拆成 near/middle/far（每点只进一个区） | — |
| 10 | `:651-671` | 分区累积 | near 3 帧 / middle 6 帧 / far 6 帧 |

阈值换算成米：

| 区 | 位置条件（x ≤16 为近） | m² | → m |
|---|---|---|---|
| Near | x ≤ 16 | 0.05 | **0.224** |
| Middle | 16 < x ≤ 22 | 0.04 | **0.200** |
| Far | x > 22 | 0.03 | **0.173** |

`params.yaml:167` 的注释是作者刻意写的：`# 三个区域与静态地图最近邻距离的平方阈值，单位 m^2。`

### 节点 3：`cluster`（283 行）

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | `:163-184` | 与 `dynamic_cloud` **完全相同的 x 分区逻辑** | 16.0 / 22.0 |
| 2 | `:225-283` | 三个区**各跑一次** `EuclideanClusterExtraction` | tol 全 0.25 |
| 3 | `:210-217` | min/max 分档 | min 5 / 3 / 3，max 5000 ×3 |
| 4 | `:186-223` | 每区质心 = 算术平均 | — |

**为什么要分区**：点云密度随距离平方衰减，远处一个目标只有几十个点、近处几千个。用同一组阈值，远处阈值太紧会碎、近处太松会并。分区是这条链路上最实用的一个补丁。

---

## 三、ROBOMASTER-HORIZON-LiDAR-2026 —— ROS 1 + 镭神 CH128X1

### `cloudCallback`（`pclmatcher.cpp:839-935`）

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | `:843` | `fromROSMsg` | — |
| 2 | `:855-872` | 三段 `PassThrough` **前置 ROI**（最省算力的位置） | x∈[-7.4,7.4]，y∈[0.5,27.5]，z∈[-3,3] |
| 3 | `:879-885` | **远处点复制 4 份**补偿稀疏（y>13） | ×4 |
| 4 | `:905-908` | 乘 `initial_alignment_transform` | — |
| 5 | `:913-922` | 攒 5 帧 → `mergePointClouds` → 交给 ICP 线程 | 5 帧 |
| 6 | `:928-931` | 转 `removeOverlappingPoints(source, field_cloud)` | — |

### `removeOverlappingPoints`（`pclmatcher.cpp:513-713`）

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | `:527-529` | `VoxelGrid` 下采样 | **0.08 m** |
| 2 | `:531-534` | `pcl::SegmentDifferences`（目标 = 背景 PCD） | **0.02 m** |
| 3 | `:589-606` | 二次 `PassThrough` 收窄 | y∈[0,15]，x∈[0,28]，z∈[-3,3] |
| 4 | `:612` | `radiusoutlierremoval(0.4, 20, 0.55, 3)` | 近(前向<13 m)：半径 0.4 / ≥20 邻居；远：0.55 / ≥3 |
| 5 | `:617` | `upsampleVoxelGrid(0.05, 3)` 体素上采样补点 | 体素 0.05，远处每体素 6 份随机抖动 |
| 6 | `:620` | `fastEuclideanCluster(0.5, 100, 7, 5000)` 自研并查集聚类 | tol 0.5，邻域上限 100，min 7，max 5000 |
| 7 | `:651-666` | 每点 `intensity = 簇ID` 发 `/obstacle_cloud` | 零成本数据契约 |
| 8 | `:675-711` | **14 m 外不做差分/聚类**，直接发 `/far_field_cloud` 给 Python 估框深 | 14 m |

**这个项目最独特的两步**：第 3 步「远处点复制 4 份」和第 5 步「体素上采样」——都是围绕**机械式 128 线在远处点太稀**做的补点，属于 Livox 不需要、但机械线雷达必须做的事。

---

## 四、TJURM-Radar —— 雷达在算法上缺席

### 离线一次性：`fuse.cpp` 建「相机背景深度图」（`main.cpp:42` 只调一次）

| 步 | 行号 | 做什么 | 参数 |
|---|---|---|---|
| 1 | `fuse.cpp:203-204` | 载入场地 PCD | 注释 FIXME：单位是 **mm** |
| 2 | `:213-220` | OpenMP 并行转 4×N 齐次矩阵 | — |
| 3 | `:~280-320` | 全部点投影到每个相机像素，收集候选深度 | — |
| 4 | `:320-322` | 按深度**升序**排序 | — |
| 5 | `:328` | z-buffer：最近面胜出 + `depth_judge(p, 3, 150)` | 3×3 邻域、±150 mm 一致性 |
| 6 | `:330-345` | 反投影得场地坐标 `depth_map[i][j].{X,Y,Z}` | — |

### 在线：`main.cpp` 用 YOLO 框查这张深度图

| 步 | 行号 | 做什么 |
|---|---|---|
| 1 | `:131-142` | YOLO 框内逐像素查背景深度图，取 3D 点 |
| 2 | `:144-172` | **真·雷达动态点云差分整段被注释**；且 `Data::kdtree` 全仓从未 `setInputCloud` |
| 3 | `:177` | `dbscan(armor_3d_point)` —— `dbscan.cpp:10-29` 实际只是求平均，没有 eps/min_samples |
| 4 | `:180` | `z >= 1100` 闸门（单位混乱） |
| 5 | `:195-196` | 「车辆去重复」= 与上一帧两点平均 |

**结论**：这条链路的「动态性先验」来自 YOLO 框（检测到装甲板 → 认为那里有动态目标），3D 深度来自**离线静态背景的深度面**，不是实时点云。雷达点云只进了共享内存没进计算。贴墙时误差是米级的。

---

## 五、四条链路并排

| 阶段 | T-DT | EGA | ROBOMASTER | TJURM |
|---|---|---|---|---|
| 前置 ROI | 变换后 x[3,28] y[0,15] z[0,1.4] | 变换后 main_roi + 4 组禁区 | **变换前**三段 PassThrough | 无（用 YOLO 框） |
| 地面处理 | **z≥0 硬切** | **z≥0.1 硬切** | **z≥-3（没切）** | 不适用 |
| 配准 | **一次性 GICP** + 0.1° 球面栅格 | 每帧 GICP→NDT0.5→NDT0.2 | 一次性 ICP（5 帧） | solvePnP（相机） |
| 差分 | K=1 平方阈值 0.1 → **0.316 m** | K=1 分区 **0.224/0.200/0.173 m** | SegmentDiff **0.02 m** | 已注释 |
| 累积 | 3 帧（统一） | **3/6/6 帧（分区）** | 5 帧（后并入 ICP） | — |
| 去噪 | 无 | 无 | **分段半径滤波 0.4/20、0.55/3** | z-buffer ±150 mm |
| 补点 | 无 | 无 | **远处 ×4 + 体素上采样 ×6** | — |
| 聚类 | tol 0.25 / min 5 / max 1000 | tol 0.25 / min 5,3,3 / max 5000 | tol 0.5 / min 7 / max 5000 | 假 DBSCAN |
| 质心 | 算术平均 | 算术平均 | 算术平均 | 算术平均 |
| 下游 | kalman 模块 | CV 卡尔曼（dt 0.0333，r 0.20 m，丢 10 帧重猜/30 帧降级） | ByteTrack + KF | 无 |

---

## 六、三条所有队都做、而当前 `car_detect` 没做的

1. **z 上下界切地面**。T-DT 用 `z∈[0,1.4]`、EGA 用 `z∈[0.1,1.3]`，**没有一个队做 RANSAC 地面分割**。在已知高度的固定安装下，一个坐标轴比较就替代了地面分割。
2. **聚类 tol 0.25 + min 3~5**。两个独立团队（其实是同源 fork）都是这个组合，ROBOMASTER 是 0.5/7。当前用 0.5/15，与任何一家都对不上。
3. **噪声交给航迹，不交给聚类**。EGA 的 min_cluster_size 低到 3，靠的是下游卡尔曼「丢 10 帧才重猜、30 帧才降级」。单帧噪点成不了航迹。

## 七、两条只有 ROBOMASTER 做、值得借鉴的

1. **`intensity = 簇ID`**（`:656`），RViz 按 intensity 着色就能直接看到「哪个点属于哪个簇」。
2. **远处点单独走另一条链**（`:675-711`）：14 m 外不做差分不做聚类，直接交给视觉估深度。承认「远处雷达不可靠」，而不是硬调参数。
