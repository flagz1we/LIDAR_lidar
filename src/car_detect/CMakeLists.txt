cmake_minimum_required(VERSION 3.8)
project(car_detect)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree search segmentation)

# ---------- pcl_process 节点 ----------
set(PCL_PROCESS_SRC src/pcl_process.cpp)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${PCL_PROCESS_SRC}")
  message(FATAL_ERROR "source file not found: ${PCL_PROCESS_SRC}")
endif()

add_executable(pcl_process ${PCL_PROCESS_SRC})

target_include_directories(pcl_process PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_compile_definitions(pcl_process PRIVATE ${PCL_DEFINITIONS})

target_link_libraries(pcl_process ${PCL_LIBRARIES})

ament_target_dependencies(pcl_process
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  tf2_eigen
  pcl_conversions
  pcl_ros
)

install(TARGETS pcl_process
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
