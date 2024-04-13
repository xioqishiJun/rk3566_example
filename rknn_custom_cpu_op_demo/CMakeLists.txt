cmake_minimum_required(VERSION 3.4.1)

project(rknn_custom_cpu_op_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# rknn api
set(RKNN_API_PATH /home/linaro/librknn_api)
set(RKNN_RT_LIB /home/linaro/librknn_api/aarch64/librknnrt.so)
set(RGA_LIB /home/linaro/librknn_api/3rdparty/rga/libs/Linux/gcc-aarch64/librga.so)



set(MPI_MMZ_LIB /home/linaro/librknn_api/3rdparty/rk_mpi_mmz/lib/Linux/aarch64/libmpimmz.so)

include_directories(/home/linaro/librknn_api/3rdparty/rk_mpi_mmz/include)
include_directories(${RKNN_API_PATH}/include)
include_directories(/home/linaro/librknn_api/3rdparty)
include_directories(/home/linaro/librknn_api/3rdparty/rga/include)
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

# install target and libraries
#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_custom_cpu_op_demo_${CMAKE_SYSTEM_NAME})
#
#install(TARGETS rknn_custom_cpu_op_demo DESTINATION ./)
#install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
#file(GLOB IMAGE_FILES "model/*.jpg")
#install(FILES ${IMAGE_FILES} DESTINATION ./model/)
#file(GLOB LABELS_FILES "model/*.txt")
#install(FILES ${LABELS_FILES} DESTINATION ./model/)
#message(STATUS "target_soc = ${TARGET_SOC}")
#install(DIRECTORY model/${TARGET_SOC} DESTINATION ./model)
