# Control layer / CLI: application entry point and orchestration.
add_executable(anpr
    src/main.cpp
)

target_link_libraries(anpr
    PRIVATE
        anpr_core
        anpr_capture
        anpr_processing
        anpr_network
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
)

# Ship the default config next to the binary so `anpr` runs out of the box.
add_custom_command(TARGET anpr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/config $<TARGET_FILE_DIR:anpr>/config
)
