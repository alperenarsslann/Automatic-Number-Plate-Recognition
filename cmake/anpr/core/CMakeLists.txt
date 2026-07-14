# Core module: shared types, interfaces, config, logging, queues.
# Every other module links against anpr_core.
add_library(anpr_core STATIC
    src/Logger.cpp
    src/Config.cpp
)

target_include_directories(anpr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(anpr_core
    PUBLIC
        opencv_core
        nlohmann_json::nlohmann_json
        Threads::Threads
)
