cmake_minimum_required(VERSION 3.4)
project(franka_interactive_controllers)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  kdl_parser
  trac_ik_lib
  trac_ik_kinematics_plugin
  controller_interface
#   passive_ds_control
  control_toolbox
  dynamic_reconfigure
  eigen_conversions
  franka_hw
  franka_gripper
  geometry_msgs
  hardware_interface
  tf
  tf_conversions
  message_generation
  pluginlib
  realtime_tools
  roscpp
  rospy
)

include_directories(/usr/local/include/qpOASES)
link_directories(/usr/local/lib)

find_package(Eigen3 REQUIRED)
find_package(Franka 0.7.0 REQUIRED)
find_package(pinocchio REQUIRED)

generate_messages()

generate_dynamic_reconfigure_options(
  cfg/minimal_compliance_param.cfg
  cfg/compliance_param.cfg
  cfg/compliance_full_param.cfg
  cfg/desired_mass_param.cfg
  cfg/gravity_compensation_param.cfg
  cfg/passive_ds_param.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES franka_interactive_controllers
  CATKIN_DEPENDS
    controller_interface
#     passive_ds_control
    control_toolbox
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    franka_gripper
    geometry_msgs
    hardware_interface
    tf
    tf_conversions
    message_runtime
    pluginlib
    realtime_tools
    roscpp
  DEPENDS Franka
)

# DECLARE LIBARY HEADER AND SOURCE FILES
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(H_FILES ${INCLUDE_DIR}/franka_cartesian_controllers/cartesian_pose_impedance_controller.h
            ${INCLUDE_DIR}/franka_cartesian_controllers/cartesian_twist_impedance_controller.h
            ${INCLUDE_DIR}/franka_cartesian_controllers/passiveDS_impedance_controller.h
            ${INCLUDE_DIR}/franka_cartesian_controllers/cartesian_pose_franka_controller.h
            ${INCLUDE_DIR}/franka_cartesian_controllers/cartesian_velocity_franka_controller.h            
            ${INCLUDE_DIR}/franka_cartesian_controllers/cartesian_force_controller.h
            ${INCLUDE_DIR}/franka_joint_controllers/joint_gravity_compensation_controller.h
            ${INCLUDE_DIR}/franka_joint_controllers/joint_impedance_franka_controller.h
            ${INCLUDE_DIR}/franka_joint_controllers/joint_position_franka_controller.h
            ${INCLUDE_DIR}/franka_joint_controllers/joint_velocity_franka_controller.h            
            ${INCLUDE_DIR}/franka_motion_generators/libfranka_joint_motion_generator.h
            ${INCLUDE_DIR}/franka_utils/pseudo_inversion.h
            ${INCLUDE_DIR}/franka_utils/kinematics_utils.hpp
            ${INCLUDE_DIR}/franka_utils/franka_ik_He.hpp
            ${INCLUDE_DIR}/franka_utils/panda_trac_ik.h)


## Specify locations of header files
## Your package locations should be listed before other locations
include_directories(include/franka_cartesian_controllers include/franka_joint_controllers include/franka_motion_generators include/franka_utils ${catkin_INCLUDE_DIRS})
include_directories(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)


## Declare a C++ executable
set(SRCS
  src/franka_cartesian_controllers/cartesian_pose_impedance_controller.cpp
  src/franka_cartesian_controllers/cartesian_twist_impedance_controller.cpp
  src/franka_cartesian_controllers/passiveDS_impedance_controller.cpp
  src/franka_cartesian_controllers/cartesian_pose_franka_controller.cpp
  src/franka_cartesian_controllers/cartesian_velocity_franka_controller.cpp
  src/franka_cartesian_controllers/cartesian_force_controller.cpp
  src/franka_joint_controllers/joint_gravity_compensation_controller.cpp
  src/franka_joint_controllers/joint_position_franka_controller.cpp
  src/franka_joint_controllers/joint_velocity_franka_controller.cpp
  src/franka_joint_controllers/joint_impedance_franka_controller.cpp
  src/franka_motion_generators/libfranka_joint_motion_generator.cpp
  src/panda_trac_ik.cpp)

add_library(franka_interactive_controllers ${H_FILES} ${SRCS})

add_dependencies(franka_interactive_controllers
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_generate_messages_cpp
  ${PROJECT_NAME}_gencpp
  ${PROJECT_NAME}_gencfg
)

target_link_libraries(franka_interactive_controllers PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
  qpOASES
  pinocchio::pinocchio
)

target_include_directories(franka_interactive_controllers SYSTEM PUBLIC
  ${Franka_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_include_directories(franka_interactive_controllers PUBLIC
  include
)

# Executable using franka_ros control interface for joint-space goal motion and open/close the gripper
add_executable(franka_gripper_run_node src/franka_gripper_run_node.cpp)
target_link_libraries(franka_gripper_run_node franka_interactive_controllers ${catkin_LIBRARIES})

add_executable(franka_joint_goal_motion_generator_node src/franka_joint_goal_motion_generator_node.cpp)
target_link_libraries(franka_joint_goal_motion_generator_node franka_interactive_controllers ${catkin_LIBRARIES})


# Executable using libfranka library ONLY for joint-space goal motion and open/close the gripper
add_executable(libfranka_gripper_run src/libfranka_gripper_run.cpp)
target_link_libraries(libfranka_gripper_run franka_interactive_controllers ${catkin_LIBRARIES})

add_executable(libfranka_joint_goal_motion_generator src/libfranka_joint_goal_motion_generator.cpp)
target_link_libraries(libfranka_joint_goal_motion_generator franka_interactive_controllers ${catkin_LIBRARIES})

add_executable(libfranka_joint_goal_motion_generator_rss src/libfranka_joint_goal_motion_generator_rss.cpp)
target_link_libraries(libfranka_joint_goal_motion_generator_rss franka_interactive_controllers ${catkin_LIBRARIES})


add_executable(libfranka_joint_goal_motion_generator_mit src/libfranka_joint_goal_motion_generator_mit.cpp)
target_link_libraries(libfranka_joint_goal_motion_generator_mit franka_interactive_controllers ${catkin_LIBRARIES})



## Installation
install(TARGETS franka_interactive_controllers
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES franka_interactive_controllers_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
catkin_install_python(
  PROGRAMS scripts/interactive_marker.py 
  scripts/move_to_start.py 
  scripts/franka_cartesian_impedance_pose_command.py
  scripts/franka_to_geometry_messages.py
  scripts/franka_gui_gripper_run.py
  scripts/spacemouse_teleop.py
  scripts/spacemouse_teleop_joint_impedance.py
  scripts/exec_policy.py
  scripts/move_gripper.py
  scripts/joint_impedance_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# install(
#   DIRECTORY teleop/
#   DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
# )

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(franka_interactive_controllers FILES ${SOURCES} ${HEADERS})
  add_tidy_target(franka_interactive_controllers
    FILES ${SOURCES}
    DEPENDS franka_interactive_controllers
  )
endif()
