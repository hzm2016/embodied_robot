cmake_minimum_required (VERSION 2.8)
# Project name
project (GravityCompensation) 

# ---- C++17 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# redis-plus-plus 没有提供 CMake config，但它安装到了 /usr/local/lib
# redis-plus-plus 在 /usr/local/include 和 /usr/local/lib
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Set directories path
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${PROJECT_SOURCE_DIR}/3rdparty/Eigen
)
# Add compiler C++11 
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")
# Add source path
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ DIR_SRCS)
# Output file
add_executable(main ${DIR_SRCS})  

target_link_libraries(main 
    PRIVATE
    redis++
    hiredis
)