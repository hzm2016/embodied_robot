cmake_minimum_required (VERSION 2.8)

project (MICRON-B_ManualCtrl)

find_path(    /usr/include/X11
              /usr/local/include
              /usr/include
    )

find_library(     /usr/lib/
                  /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                  /usr/local/lib64/
                  /usr/local/lib/
                  /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
    )

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/rci_sdk1.4.0/lib)

aux_source_directory(./src/ DIR_SRCS)

add_executable(MB_CTRL ${DIR_SRCS})

include_directories (
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    /usr/include/eigen3 
    rci_sdk1.4.0/inc 
    rci_sdk1.4.0/inc/include 
    rci_sdk1.4.0/inc/rci_data  
)

target_link_libraries(MB_CTRL
/usr/lib/libHD.so libHL.so
/usr/lib
/usr/local/lib
${DEPENDENCIES}
RCI_Client.a
xMateModel.a
orocos-kdl.a
glog.a
pthread
    )
