# Mock site-server: standalone TCP test tool for the network layer.
add_executable(mock_server mock_server.cpp)

target_link_libraries(mock_server PRIVATE nlohmann_json::nlohmann_json)

if (WIN32)
    target_link_libraries(mock_server PRIVATE ws2_32)
endif()
