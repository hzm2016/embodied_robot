cmake_minimum_required (VERSION 2.8)
# Project name
project (GravityCompensation)

find_path(HIREDIS_INCLUDE_DIR
    NAMES hiredis/hiredis.h
    REQUIRED
)

find_library(HIREDIS_LIB
    NAMES hiredis
    REQUIRED
)
# Set directories path
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${PROJECT_SOURCE_DIR}/3rdparty/Eigen
    /usr/local/include/
    ${HIREDIS_INCLUDE_DIR}
)

# Add compiler C++11 
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++17 -pthread")
# Add source path
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ DIR_SRCS)
# Output file
add_executable(main ${DIR_SRCS})

# 动态库绝对路径（根据实际文件名与路径修改）
set(HANS_SDK_SO ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libHR_Pro.so)

target_link_libraries(main PRIVATE ${HANS_SDK_SO} ${HIREDIS_LIB})

set_target_properties(main PROPERTIES
BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/hans/lib" 
INSTALL_RPATH "$ORIGIN/../lib" ) 

