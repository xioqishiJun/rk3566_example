cmake_minimum_required(VERSION 3.4.1)

project(rknn_custom_cpu_op_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 获取HOME环境变量
set(HOME_DIR $ENV{HOME})

# 打印HOME环境变量
message("Home directory is: ${HOME_DIR}")

# rknn api
set(RKNN_API_PATH ${HOME_DIR}/librknn_api)
set(RKNN_RT_LIB ${HOME_DIR}/librknn_api/aarch64/librknnrt.so)
set(RGA_LIB ${HOME_DIR}/librknn_api/3rdparty/rga/libs/Linux/gcc-aarch64/librga.so)



set(MPI_MMZ_LIB ${HOME_DIR}/librknn_api/3rdparty/rk_mpi_mmz/lib/Linux/aarch64/libmpimmz.so)

include_directories(${HOME_DIR}/librknn_api/3rdparty/rk_mpi_mmz/include)
include_directories(${RKNN_API_PATH}/include)
include_directories(${HOME_DIR}/librknn_api/3rdparty)
include_directories(${HOME_DIR}/librknn_api/3rdparty/rga/include)
set(CMAKE_INSTALL_RPATH "lib")


#add_executable(rknn_custom_cpu_op_demo
#  src/rknn_api_test_custom_cpu_op.cpp
#)
add_executable(rknn_custom_cpu_op_demo
  src/main.cpp
)
target_link_libraries(rknn_custom_cpu_op_demo
  ${RKNN_RT_LIB} ${RGA_LIB} ${MPI_MMZ_LIB}
)

