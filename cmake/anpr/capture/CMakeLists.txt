# Capture & decode layer: IFrameSource implementations + factory.
add_library(anpr_capture STATIC
    src/SimulatedFileFrameSource.cpp
    src/HikvisionRtspFrameSource.cpp
    src/HikvisionDiscovery.cpp
    src/FrameSourceFactory.cpp
)

target_include_directories(anpr_capture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(anpr_capture
    PUBLIC
        anpr_core
    PRIVATE
        opencv_videoio
)

if (WIN32)
    target_link_libraries(anpr_capture PRIVATE ws2_32 iphlpapi)
endif()
