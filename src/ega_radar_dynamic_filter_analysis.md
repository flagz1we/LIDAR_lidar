# EGA_Radar_Algorithm 动静剔除 / 动目标提取 分析报告

路径前缀约定：`$P = /home/tjurm/Code/radar_projs/EGA_Radar_Algorithm-main`

---

## 0. 结论先行

**这个项目确实做了激光雷达动静剔除，而且它是整条雷达链路的第一步。** 但它用的**不是**占据栅格、不是帧间差分、不是光线投射自由空间剥离、不是深度图 MOG2、也**不是**跟踪速度阈值，而是最朴素的一种：

> **静态先验地图差分（prior-map differencing）** —— 把离线建好的场地 PCD 建 KD-Tree，实时点云转到场地坐标系后逐点查最近邻，**平方距离超过阈值就判为动点**。

一句话公式（`$P/Lidar_ws/src/dynamic_cloud/src/dynamic_cloud.cpp:466`）：

```text
dynamic  ⟺  ‖p − NN_map(p)‖²  >  thr(region(p))
```

配套的第二个机制是 `localization` 里的**角度栅格持久背景（angular-bin persistent background）**，但它**只用于给配准器提供 source 点云**，不产生动目标输出（详见 §2）。

| 问题 | 答案 |
|---|---|
| 主算法 | 静态地图差分 + 分区阈值（近/中/远三档） |
| 栅格法 | 否（背景角度栅格是配准用的，非目标输出） |
| 帧间差分 | 否 |
| 光线投射/自由空间 | 否（项目里的 raycast 是相机射线×PLY 求交，与雷达无关） |
| 速度阈值 | 否（卡尔曼速度不参与动静判定） |
| 传感器 | Livox HAP 固态非重复扫描 |
| 聚类 | PCL 欧式聚类，容差 0.25 m，输出质心 |
| 跟踪 | 有，但纯 2D 场地平面 KF，在下游 Python 侧 |

---

## 1. 动静分离的具体算法

### 1.1 主链路（唯一产生动目标的地方）

`dynamic_cloud` 节点，函数 `GetDynamicCloud()`，`$P/Lidar_ws/src/dynamic_cloud/src/dynamic_cloud.cpp:438-480`。

关键 12 行（`dynamic_cloud.cpp:459-469`）：

```cpp
std::vector<int>   pointIdxNKNSearch(K);            // K = 1
std::vector<float> pointNKNSquaredDistance(K);
if (kd_Tree.nearestKSearch(input_cloud.points[j], K,
                           pointIdxNKNSearch,
                           pointNKNSquaredDistance) > 0) {
    const float threshold = static_cast<float>(
        GetMapDistanceThreshold(input_cloud.points[j]));   // 按区域取阈值
    if (pointNKNSquaredDistance[0] > threshold) {
        clouds[i].points.push_back(input_cloud.points[j]);  // → 动态点
    }
}
```

要点：

- **K = 1**，只查最近一个地图点（`:442`），不做 k 邻域平均、不做平面拟合残差。
- 判定为**严格的单边门限**：`> thr` 保留为动点，`≤ thr` 直接丢弃。没有任何 hit/miss 计数、没有概率、没有衰减。
- 地图 KD-Tree **只在构造时建一次**（`:355` `kd_Tree.setInputCloud(map_cloud)`），**运行期永不更新**。这一点决定了后面的所有行为（见 §7）。
- 多线程：按点数切块，`dynamic_thread_num: 12` 个 `std::thread`（`:453-472`，参数见 `params.yaml:174`）。

### 1.2 前置的坐标变换与 ROI 硬裁剪

动点提取之前有三道空间过滤（`dynamic_cloud.cpp:567-609`）：

1. **TF 变换到 `rm_frame`**（`:568-586`）——查 `rm_frame` ← `msg->header.frame_id` 的变换，失败则**整帧丢弃**（`:571-575`）。这就是全系统对定位的硬依赖。
2. **固定干扰区直接删除**（`:592-598`）——飞镖区、基地、斜矩形区、若干轴对齐区。注释原文：`// 地图中固定干扰区域直接剔除，避免进入动态点和告警检测。`
3. **主场地区域 + 高度门限**（`:599-609` + `params.yaml:181`）——`main_roi = [0.3, 27.4, 0.3, 14.7, 0.1, 1.3]`，即只在 `0.1 m < z < 1.3 m` 的高度带里找目标，地面和高于 1.3 m 的结构全部不看。

### 1.3 明确否定的几种方案

我逐个关键词查过，以下机制**在整个仓库中不存在于雷达链路**（`model/` 下的 YOLO 训练代码除外）：

- `occupancy` / `OccupancyGrid` / 占用栅格计数 → 无
- `frame_diff` / 帧间差分 → 无
- `MOG2` / `createBackgroundSubtractor` / 深度图背景建模 → 无
- `raycast` / `free_space` 自由空间剥离 → 无。项目里的 `ray` 全是**相机射线与 `RMUC2026.ply` 三角网格求交**，用于把像素投影到场地平面（`$P/Main_ws/transform/ray_renderer.py`、`xy_map.py`），与雷达动点提取无关。
- 速度阈值 → 无。卡尔曼速度 `vx, vy` 只用于预测位置延续，**不参与任何动静判定**（`$P/Main_ws/tracker/CascadeMatchTracker/tracker.py` 中 `velocity` 只出现在初始化 `initial_velocity_std_mps`，`:505`）。
- `counter` / `lifetime` / `hit` / `miss` / `decay` 栅格计数器 → 无。唯一出现的 `decay_tau`（`params.yaml:86`）是**视觉 BoT-ID 语义历史**的时间衰减，与点云无关。

---

## 2. 栅格法？—— 有一个角度栅格，但不是用来出目标的

`localization` 节点里有一套**方位角/俯仰角栅格 + 最远点优先**的持久背景，用于压制"近处机器人遮挡"对**配准**的干扰。它**不输出动目标**，只作为配准 source。

| 项目 | 值 | 位置 |
|---|---|---|
| 是否启用 | 是（`enable_background_filter: true`） | `params.yaml:57` |
| 栅格分辨率 | **0.1°**（方位角 × 俯仰角） | `params.yaml:59` |
| 每格存什么 | 该角度格内**最远**的那个点 | `localization.cpp:607-635` |
| "命中"处理 | 新点更近 → 视为前景遮挡，**丢弃**；新点略远（≤0.10 m）→ 保留更远点 | `localization.cpp:708-716` |
| "未命中"处理 | 新点显著更远（> 0.10 m）→ 进**候选**队列，需累计确认 | `localization.cpp:718-736` |
| 候选确认次数 | `background_candidate_confirm_count: 1` | `params.yaml:63` |
| 候选一致性容差 | `background_candidate_tolerance: 0.15` m | `params.yaml:65` |
| 距离容差 | `background_range_tolerance: 0.10` m | `params.yaml:61` |
| **衰减周期** | **无衰减**。没有任何时间/帧计数递减机制 | — |
| "永久静态"判定阈值 | **无**。背景一旦写入就是永久的，只有 reset 才清空 | — |

**所以：如果按"占据栅格 + hit/miss 计数 + 衰减 + 终身值"这套 SLAM 语境来问，答案是没有。** 这套角度栅格本质是"逐角度保留最远回波"，是一个**遮挡抑制滤波器**，跟 occupancy grid 的语义完全不同。

栅格位置定义在 `localization.cpp:621-629`：

```cpp
const double azimuth   = std::atan2(point.y, point.x);
const double elevation = std::atan2(point.z,
                          std::sqrt(point.x*point.x + point.y*point.y));
const int azimuth_index = static_cast<int>(
    std::floor(azimuth * 180.0 / M_PI / angular_bin_deg));
```

背景的自动重建条件（判定"雷达被移动了"）：当前帧与 old 的**内点率**低于 `registration_reset_inlier_ratio_threshold: 0.90`，且双方各有点数 ≥ `registration_reset_min_points: 200`，且背景年龄 ≥ `10` 帧（`params.yaml:72-78`，实现 `localization.cpp:661-694`、`:789-804`）。内点判据：到 old 最近点距离 ≤ `0.10 m`（`params.yaml:72`）。

**注意**：这套背景结果只发布到 `/filter_map`（`localization.cpp:514`、`:875`），全仓库唯一订阅它的是…没有人（`/livox/map` 出现在 RViz 配置 `:110`，`/filter_map` 连 RViz 都没订阅）。**纯调试话题，不进主链路。**

---

## 3. 静态地图差分：地图、配准、阈值

### 3.1 地图怎么生成

- 文件：`$P/Lidar_ws/src/dynamic_cloud/config/RMUC2026.pcd`
- 实测头部：`WIDTH 358525` / `POINTS 358525` / `DATA ascii` / 31 MB，字段 `x y z` + `rgb normal_x normal_y normal_z Coord._Z`
- 由**同一台 Livox 雷达对空场地扫描 + 离线拼接**得到，不是 CAD 导出。配置注释里透露了手工修图的痕迹：`map_exclude_rotated_roi` 的值来自 `RMUC2026_picked_points.pp` 四点拟合（`params.yaml:186`）——即用 CloudCompare 手工挑点拟合的斜矩形。
- 加载后**立即体素下采样**再建树（`dynamic_cloud.cpp:347-355`）：`voxel_leaf_size: 0.1` m（`params.yaml:146`）。

### 3.2 怎么配准

配准不在 `dynamic_cloud` 里做，而是由 `localization` 节点做完整 3D 配准后**通过 TF 发布**，`dynamic_cloud` 只负责消费 TF。

- 后端 `backend: "gicp_ndt"`（`params.yaml:36`）：**GICP → 粗 NDT(0.5 m 体素) → 细 NDT(0.2 m 体素)** 三段串联（`params.yaml:117-128`）
- GICP：`max_iteration 10`、`max_correspondence_distance 2.0 m`、`euclidean_fitness_epsilon 0.0001`、`rotation_epsilon 0.002`（`params.yaml:103-115`）
- 粗 NDT：`resolution 0.5`、`max_iteration 30`、`step_size 0.1`；细 NDT：`resolution 0.2`、`max_iteration 30`、`step_size 0.05`；`outlier_ratio 0.35`、`neighborhood_search "direct7"`
- 源/目标体素：均 `0.1 m`（`params.yaml:139-142`）
- 初始位姿来自外参 `lidar_extrinsics.yaml`：`translation [0.0, 6.5, 3.8]`、`rpy [0.0, 10.0, 0.0]`（`:8-9`）
- 位姿后处理：`fitness_threshold 0.01`、`stable_required_count 30`、`max_translation_jump 10.0 m`、`max_rotation_jump_deg 20.0`、`transform_smooth_alpha 0.35`（`params.yaml:86-101`）
- 配准频率被限流：`registration_interval_sec: 0.5`（≈2 Hz），且 `max_registration: 1` —— **默认只配准一次就锁定位姿**（`params.yaml:48-51`）。这对动静剔除是很关键的设计选择（见 §9）。

### 3.3 差分阈值多大

阈值**是平方距离，单位 m²**，这是最容易踩的坑。配置文件写得很明确（`params.yaml:167-171`）：

```yaml
# 三个区域与静态地图最近邻距离的平方阈值，单位 m^2。
# 影响：点到地图最近邻平方距离大于所在区域阈值时认为是动态点。
near_map_distance_threshold: 0.05
middle_map_distance_threshold: 0.04
far_map_distance_threshold: 0.03
```

换算成**米**：

| 区域 | 平方阈值 (m²) | 等效距离 (m) | 场地范围 (x) |
|---|---:|---:|---|
| 近端 Near | 0.05 | **0.224** | x ≤ 16.0 |
| 中间 Middle | 0.04 | **0.200** | 16.0 < x ≤ 22.0 |
| 远端 Far | 0.03 | **0.173** | x > 22.0 |

分区规则（`dynamic_cloud.cpp:382-403`）：`region_split_axis: "x"`、`near_middle_split_position: 16.0`、`middle_far_split_position: 22.0`、`near_region_positive: false`（`params.yaml:160-166`。雷达在 `x=0`，所以 x 小的一端是近端）。

**方向值得注意**：距离越远，阈值越**紧**（0.224 → 0.200 → 0.173 m），与"远处点稀、应放宽容差"的直觉相反。文档给的理由是"以适应不同距离下的点云密度和测量误差"（`$P/docs/融合定位算法说明.md:93`）。工程上合理的解读是：远端机器人只有零星几个点，阈值一旦放宽，这些点会被附近的墙面/地面地图点"吸"成静态而丢失；宁可远端残留更多静点，也不能丢掉仅有的动点。代码本身没有解释这个方向性，这是**我的推断**，不是原文。

---

## 4. 输入点云：传感器、线数、帧率、每帧点数

| 项目 | 值 | 依据 |
|---|---|---|
| 型号 | **Livox HAP**（固态非重复扫描，非机械多线） | `$P/README.md:23`；`$P/docs/主程序数据流说明.md:39` |
| 话题 | `/livox/lidar` | `dynamic_cloud.cpp:358` |
| 驱动 | 外部 `livox_ros_driver2`，launch 为 `launch_ROS2/pcd_HAP_launch.py` | `params.yaml:28` |
| 安装位姿 | 场地中线旁，`[x,y,z] = [0.0, 6.5, 3.8] m`，`pitch 10°` | `lidar_extrinsics.yaml:8-9` |
| 点类型 | 转成 `pcl::PointXYZ`，**丢弃 intensity / tag / return** | `dynamic_cloud.cpp:561` |
| 线数 | **不适用**——HAP 是面阵式非重复扫描，无"线"概念 | — |
| 帧率 | **仓库未声明** | 见下 |
| 每帧点数 | **仓库未声明** | 见下 |

关于帧率与点数的诚实说明：**配置文件里没有任何 `frame_rate` / `points_per_frame` 字段**，rosbag 元数据也未纳入 Git（`README.md:126`：回放数据走网盘分发）。唯一可用的间接证据是 `README.md:120` 的记录统计：

> 完整录制包含 17,028 帧原分辨率相机图像、12,828 帧 `/livox/lidar` 点云

配合 `README.md:112` 的"约 9.9 分钟有效同步数据"，可反推：`12828 / (9.9×60) ≈ 21.6 Hz`，相机侧 `17028 / 594 ≈ 28.7 fps`。**所以雷达约 20 Hz 量级是推算值，不是文档值。** 每帧点数无法从仓库推算（原始 bag 不在仓库内）。唯一出现的 `max_points_per_frame: 10000`（`Main_ws/config/device.yaml:32`）经全仓 grep 确认**在 Python 侧根本没有被读取**，是个死配置项，不能当作每帧点数。

一个可参考的规模感：静态地图 358,525 点（下采样前），配准 ROI 为 `x∈(5,30)`, `y∈(-10,8)`, `z<7`（`localization.cpp:568-571`）。

---

## 5. 剔除完之后怎么聚类、参数多少

`$P/Lidar_ws/src/cluster/src/cluster.cpp`，标准 PCL 欧式聚类：

```cpp
pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;   // :199
ec.setClusterTolerance(params.tolerance);            // :200
ec.setMinClusterSize(params.min_size);               // :201
ec.setMaxClusterSize(params.max_size);               // :202
```

输入是**多帧累积后的动点云**（`/livox/lidar_dynamic`，`:141-143`），输出是**每个簇的几何质心**（不是包围盒、不是完整点云，`:208-222` 求平均），发布到 `/livox/lidar_cluster`（`:147-148`）。

聚类同样按近/中/远分区独立执行（`:247-267`），参数（`params.yaml:211-219`）：

| 区域 | 容差 (m) | min points | max points |
|---|---:|---:|---:|
| Near | 0.25 | **5** | 5000 |
| Middle | 0.25 | **3** | 5000 |
| Far | 0.25 | **3** | 5000 |

**多帧累积参数**（这是动静剔除之后的第二级"提纯"，`params.yaml:150-158`）：

| 区域 | 累积帧数 | 设计意图（原注释） |
|---|---:|---|
| Near | **3** | "近端点密度较高，较小值可缩短运动残影和聚类中心滞后" |
| Middle | **6** | "在近端响应速度和远端点云密度之间折中" |
| Far | **6** | "较大值可补充远端稀疏点，但过大会加重高速目标拖影" |

缓存实现是 `std::deque`，每帧 push_back 并按 `max_frames` 裁掉队首（`dynamic_cloud.cpp:419-427`）。注意 `accumulate_time` 的单位是**帧数不是秒**（`params.yaml:30-32` 明确写了"代码按帧计数，并不是秒"）。

区域缓存的一个细节值得抄（`dynamic_cloud.cpp:614-615`）：

> `// 每个点只进入一个区域缓存；历史缓存可保证目标跨越分界时平滑过渡，`
> `// 不在边界重复点，避免人为增加聚类密度。`

即累加和去重是**在分区之后**做的，而不是把原始帧先合并再分类。

---

## 6. 跟踪 / 卡尔曼 / 航迹管理

**有，但完全在下游 Python 侧，且是纯 2D 场地平面滤波，不参与动静剔除。**

- 实现：`$P/Main_ws/tracker/CascadeMatchTracker/kalman.py:90-160`
- 模型：**常速度 CV 模型**，状态 `[x, y, vx, vy]`，量测 `[x, y]`
- 参数（`params.yaml:116-124`）：

| 参数 | 值 |
|---|---|
| `default_dt_sec` | 0.0333 s |
| `min_dt_sec` / `max_dt_sec` | 0.001 / 0.20 s（dt 钳位） |
| `process_q_std` | 2.0 |
| `lidar_r_std_m` | **0.20 m**（雷达量测噪声） |
| `initial_position_std_m` | 0.25 |
| `initial_velocity_std_mps` | 2.0 |
| `hard_reset_distance_m` | **2.5 m**（新量测与预测差超过此值则重初始化，避免旧轨迹拖拽新目标） |

Q 矩阵按 `q_std² × [[dt⁴/4, dt³/2],[dt³/2, dt²]]` 构造（`kalman.py:126-134`），P0 = `100·I`。`filterpy` 缺失时有一个 Joseph 形式的 numpy 兜底实现（`kalman.py:5-36`）。

**航迹管理**（`params.yaml:126-128`）：

| 参数 | 值 | 行为 |
|---|---|---|
| `radar_guess_after_frames` | **10** 帧 | 允许用未分配的剩余雷达点重新猜测位置 |
| `fixed_guess_after_frames` | **30** 帧 | 退化为 `config/guess_pts.yaml` 里的固定候选点 |

**关联方式**（级联，按序抢占，`docs/融合定位算法说明.md:181-189`）：KF 预测 → 己方哨兵反馈剔除 → 彩色装甲视觉-雷达三元组匹配（门限 `1.5 m`，`params.yaml:91`）→ 灰色装甲模糊匹配 → 历史预测位置↔剩余雷达点匈牙利匹配（门限 `2.0 m`，`params.yaml:105`）。

另有 `bot_history` 语义历史（`queue_size 30`、`matching_window 10`、`decay_tau 0.5`、`association_ttl_frames 30`，`params.yaml:84-88`）。

**关键点**：以上全部工作在"动点已经提完"之后。整个跟踪器没有反向影响动静判定——没有"确认航迹→把该位置标为动区"这类回灌机制。

---

## 7. 「静止但应当保留」的特殊处理

这是本项目最值得注意的一个设计后果，分两面说。

### 7.1 静止的车/机器人：**自然被保留，因为地图永不自更新**

因为 KD-Tree 是**启动时一次性建的、运行期完全不更新**（`dynamic_cloud.cpp:344-355`），所以：

> 一个静止的敌方机器人**不在先验地图里** → 它的点永远满足 `d² > thr` → **永远被判为动点并持续输出**。

这正好满足"静止的车应当保留"的需求，而且**零额外代码**。这是先验地图法相对"背景学习/占据栅格"方案的一个天然优势：背景学习法会把停车超过若干帧的目标吸收进背景而丢失，本项目没有这个风险。

代价是**对称的恶果**（见 §9）：任何**地图里没有的静止物**（倒地机器人、场地上的杂物、站着的人、新摆放的道具）都会被**永久**判为动目标，且因为没有任何衰减机制，它永远不会被自动抑制。

### 7.2 静止干扰点：靠**手写 ROI 黑名单**硬删

实际场地上有大量"应该删掉的静物"，但算法层面无法区分，于是作者选择了**人工标注删除区**。这类参数共 5 组（`params.yaml:183-194`）：

| 参数 | 值 | 说明（原注释） |
|---|---|---|
| `dart_exclude_roi` | `[26.0, 28.0, 3.0, 5.5, 0.0, 2.5]` | "飞镖固定剔除区域，区域内点直接丢弃" |
| `home_exclude_roi` | `[26.5, 28.0, 13.0, 15.0, 0.0, 3.0]` | "基地区域固定剔除区域" |
| `map_exclude_rotated_roi` | `[14.0, 7.5, 45.3, 3.0, 1.7, -inf, inf]` | 斜矩形，来自 `RMUC2026_picked_points.pp` 四点拟合 |
| `map_exclude_axis_aligned_rois` | 4 个轴对齐盒子 | "国赛实际场地和建模时不同的区域" |

处理位置在 `dynamic_cloud.cpp:592-598`，是**在查 KD-Tree 之前**就把点扔掉，注释写得很直白：`// 地图中固定干扰区域直接剔除，避免进入动态点和告警检测。`

第 4 组（`params.yaml:190-194` 的四个盒子：`[24.8,26.3]×[7.0,8.0]`、`[24.9,inf]×[0.0,3.4]`、`[22.8,23.8]×[4.70,5.05]`、`[21.1,21.9]×[10.7,11.3]`）是**最能说明问题的证据**：这些是"场地实物与 PCD 模型不一致"的位置，差分法在这些地方会持续吐出假动点，作者只能逐个量出坐标手写进配置。

### 7.3 己方机器人本体

雷达站自身和己方哨兵的处理**不靠几何**，而是靠**裁判系统反馈**：`tracker.py:1041` 的模块 `"10.4 己方哨兵反馈匹配与雷达点剔除"`，用新鲜的己方哨兵位置匹配并剔除对应雷达点。参数 `ally_fusion.feedback_timeout_sec: 0.15`、`max_lidar_match_distance_m: 1.5`（`params.yaml:106-112`）。没有反馈时该机制失效，己方机器人会被当成普通动点。

### 7.4 高度门限

`main_roi` 的 `z ∈ (0.1, 1.3) m` 同时承担"排除地面"和"排除场馆高结构"两个职责，是个很粗但有效的静物过滤。

---

## 8. 代码里关于「为什么这么做」「踩过什么坑」的注释

**有，而且密度不低。** 主要分布在 `params.yaml`（几乎每个参数都有"影响/说明"两行）和 `localization.cpp`，以及 `docs/`。原文摘录：

### 8.1 关于动静差分本身

`$P/Lidar_ws/src/rm2026_lidar_launch/config/params.yaml:167-168`
```
# 三个区域与静态地图最近邻距离的平方阈值，单位 m^2。
# 影响：点到地图最近邻平方距离大于所在区域阈值时认为是动态点。
```
→ 作者**清楚**自己在比较平方距离，并主动在单位里写 `m^2`。这不是 bug，但对外部使用者是最大的陷阱（把 0.05 当成 5 cm 会差 4 倍）。

`$P/Lidar_ws/src/dynamic_cloud/src/dynamic_cloud.cpp:614-615`
```
// 每个点只进入一个区域缓存；历史缓存可保证目标跨越分界时平滑过渡，
// 不在边界重复点，避免人为增加聚类密度。
```
→ 坑：如果先累积再分区，跨界的点会被两侧各算一次，聚类密度虚高。

`params.yaml:148`
```
# 影响：不再直接决定地面机器人点云累积，地面三个区域分别使用下面的参数。
```
→ 历史痕迹：原来只有一个 `accumulate_time`，后来拆成三档，旧参数降级为"飞镖/飞行器"专用 + 兼容回退。

### 8.2 关于持久背景（配准侧）

`params.yaml:53-55`
```
# true：始终按角度栅格逐帧更新old，近处动态遮挡不会覆盖old；
# 配准锁定、达到max_registration或NDT达到max_iteration均不停止old更新。
```
→ 坑：把背景更新挂在"配准是否进行"上，一旦配准锁定背景就冻结，机器人挡住的地方会永久污染背景。

`localization.cpp:656-657`
```
// 角度栅格会跨帧累积；再次使用与单帧筛选相同的体素尺寸，
// 保证最终发布并送入配准器的old不会重新变得过密。
```
→ 坑：栅格累积会让点云越积越密，配准耗时爆炸。重复体素下采样解决。

`localization.cpp:806-807`
```
// old的维护独立于registration状态；配准锁定、达到次数上限或
// NDT阶段达到迭代上限，都不应阻止背景继续吸收后续帧。
```
`params.yaml:62`
```
# 新点显著远于old时，需要多少次距离一致的观测才允许替换old，避免一次远距离噪点污染背景。
```
`params.yaml:77`
```
# old初始化/重置后至少经过这么多帧，才开始执行移动判定，避免背景尚未建立时误重置。
```
→ 两个对称的坑：**替换太激进** → 一个远距离噪点永久污染背景；**判定太激进** → 背景还没建好就误判"雷达被移动了"然后无限重置。

`localization.cpp:853-854`
```
// latest-only mailbox：worker忙时不积压队列，新帧直接覆盖尚未执行的任务。
// registration次数和初值在worker真正取走任务时确定，覆盖不会消耗次数。
```
→ 坑：配准线程慢于点云帧率时的队列积压/次数误计。

### 8.3 关于累积与聚类

`$P/docs/融合定位算法说明.md:104`
```
多帧累积能提高聚类稳定性，但目标高速运动时会产生一定拖尾，因此聚类中心并不等同于机器人几何中心。
```
→ 明确的已知缺陷声明。

### 8.4 关于配准后端选择（这段是最好的"踩坑史"）

`$P/docs/融合定位算法说明.md:51-53`
```
GICP 经过大量实验能够提供了良好的初步位姿，但由于基于硬匹配的优化函数性质不够良好，
匹配过程总会收敛在和实际点云平均相差10-20cm的位置。
NDT 基于局部概率密度求取极大似然，其性质极其良好，但细粒度的 NDT 依赖初值。
注：也可以采用极大粒度的 NDT 依次降粒度。但由于前期一直使用 GICP ，故为了比赛稳定性仍然保留使用 GICP 提供初值。
```
→ **这段和动静剔除直接相关**：GICP 残差 10–20 cm，而动点门限是 17.3–22.4 cm。也就是说**位姿误差几乎吃掉了整个门限余量**。这是全方案最脆的一环。

### 8.5 关于融合架构（同样是"妥协"记录）

`docs/融合定位算法说明.md:168`
```
注：原本希望使用激进的(历史, 视觉, 雷达)三元组的融合匹配。但是限于开发时还没有足够好的数据难以调试，
故退而使用稳定性更高的分阶段匹配。由于信息波解得过于稳定，所以直接使用信息波，而不考虑融合信息波。
```
`docs/融合定位算法说明.md:206`
```
注：后续发现，训练好的模型在合适的阈值下并没有出现多个视觉框指向同一个 id 的情况，
故实际情况恰好退化为了二元组，可以改为匈牙利匹配。但由于其本身时间代价已经极小，为了稳定性仍保留。
```

### 8.6 诚实的遗留标记

`params.yaml:195, 198, 207` 三处仍留着：
```yaml
# main_roi: [...] # TOBE CHANGED
# dart_roi: [...] # TOBE CHANGED
# fly_detect_roi: [...] # TOBE CHANGED
```
→ 作者自己标了"待改"，没有假装完稿。

---

## 9. 这个方案的优缺点与适用条件

### 优点

1. **极简、可预测、零训练**。全链路就是 KD-Tree 一次最近邻查询，没有概率、没有状态机、没有需要调的超参收敛问题。`dynamic_cloud.cpp:438-480` 一共 43 行，任何人 10 分钟能读懂并改。
2. **静止目标永不丢失**。地图不自更新的特性，天然满足"停车/停走的目标必须保留"这个 RoboMax 雷达站的核心需求（§7.1）。这是它相对 background-learning 类方案的决定性优势。
3. **分区调参把距离维度的问题工程化解决了**。近/中/远三档阈值 + 三档累积帧数（0.224/0.200/0.173 m，3/6/6 帧），用 4 个数字覆盖了"近处点密会拖影、远处点稀会丢点"这对矛盾。这种"分档代替建模"的做法在比赛场景里性价比极高。
4. **算力友好**。12 线程按点数切块（`dynamic_cloud.cpp:453-472`），单次查询 O(log N)，358k 点地图的 KD-Tree 完全不是瓶颈。整个节点能塞进 intra-process composable container（`dynamic_cloud/launch/lidar.launch.py`）。
5. **失败模式可观测**。所有参数都有 `RCLCPP_WARN` 回退（`dynamic_cloud.cpp:242-341` 一大段），非法值不会静默崩，而是打日志 + 用默认值继续跑。比赛场景这是正确的取舍。

### 缺点

1. **门限余量被位姿误差吃干**。门限 0.173–0.224 m，而配准残差 10–20 cm（`docs/融合定位算法说明.md:51`），两者同量级。结论：**定位一差，所有静态表面立刻糊出一层假动点壳**，表现为"墙边一直有动目标"。没有 hit/miss 计数或形态学滤波来兜底，误检会直接进聚类和下游关联。
2. **无法处理"地图没有的静物"**。倒地机器人、场地杂物、站立的裁判，一旦静止就**永久**是动目标。系统里没有任何衰减、没有终身值、没有二次确认。唯一的对抗手段是手写 ROI 黑名单（§7.2），这意味着**换一个场地就要重新量坐标改配置**。
3. **对场地建模精度零容忍**。`map_exclude_axis_aligned_rois` 的四组坐标（`params.yaml:190-194`）就是"模型和实物不一样"的血泪账。这类硬编码不可迁移、不可自愈。
4. **多帧累积与动态性天然冲突**。3/6/6 帧累积（20 Hz 下约 0.15–0.3 s）在高速目标上必产生拖尾，作者自己承认"聚类中心并不等同于机器人几何中心"（`docs:104`）。对需要精确几何中心的上层算法（自瞄、预测）是隐患。
5. **输出信息被压扁到质心**。聚类只发质心（`cluster.cpp:208-222`），下游再也拿不到点数和形状，无法在高层用"点数突增说明目标靠近"这类判据。`min_cluster_size` 是唯一的点数过滤，且近端=5 相当宽松（20 Hz 累积 3 帧后，一个远端机器人轻松过 5 点）。
6. **两套"背景"概念容易混淆维护**。`dynamic_cloud` 的静态地图（管目标输出）和 `localization` 的角度栅格背景（管配准）是两套完全独立的东西，参数名都以 `background_*` / `map_*` 出现，且后者结果只发到没人订阅的 `/filter_map`。新人极易改错地方。
7. **`max_registration: 1` 的隐含风险**。默认只配准一次就锁定（`params.yaml:51`），配合 `registration_reset_auto: true` 才有重入机会。如果雷达被轻微碰动且内点率仍在 0.90 以上，系统会带着错误位姿一直跑到结束。

### 适用条件

**这套方案适合：**

- 场地**固定且可提前离线精细建模**，且建模后可保证实物不再改变（RoboMax/RMUC 标准赛场就是这种）。
- 雷达**安装位置固定不动**（本项目外参写死在 `lidar_extrinsics.yaml:8-9`，`translation [0, 6.5, 3.8]`）。
- 待检测目标**必然运动**，或"静止但地图里没有"（这正是雷达站关心的敌方机器人）。
- 有**可靠的 3D 配准**把位姿误差压到**明显小于动点门限**（本项目实测没做到，10–20 cm vs 17–22 cm，属勉强）。**若要用在自己项目上，第一优先级是把配准残差压到门限的 1/3 以下，即 ≤5–7 cm。**
- 算力有限、要求延迟确定、不接受训练/在线学习。

**这套方案不适合：**

- 场地会变（临时赛场、练习场地、道具会移动）→ 会持续误检，只能靠不断加 ROI 黑名单续命。
- 雷达会被移动或重新架设 → 每次都要重建 PCD 或至少重标外参。
- 存在大量"静止但非地图"的物体（人员走动后停留、器材堆放）。
- 需要目标几何尺寸/朝向（只给质心）。
- 需要区分"慢速运动"与"静止"（本方案对速度完全不敏感，1 cm/s 和 10 m/s 的点一视同仁，全靠地图有没有）。

### 如果要在自己项目上借鉴，建议的三条改进

1. **保留"固定先验地图 + 永不自更新"这个核心决策**——它解决了"停车目标不能被吸收"这个最难的问题，是这套方案最值钱的地方。
2. **把单边硬门限换成带滞回的软判据**：对地图差分残差做一次小连通域分析，去掉孤立点（当前的 KD-Tree 单点判定对噪点毫无抵抗力），再对"连续 N 帧在同一栅格被判为动点"的位置打标记做抑制——这正是本项目缺的 hit/miss 层，能补掉缺点 2 的大部分。
3. **把 `map_distance_threshold` 从"平方值"改成"米"**。当前配置写成 `0.05` 而语义是 `0.05 m²`（`params.yaml:167`），虽然作者有意为之并留了注释，但这是极易被后人误读的接口，改名为 `*_map_distance_sq_threshold` 或直接存米都更安全。
