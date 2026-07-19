# Network / API layer: INetworkTransport implementations + message schema.
add_library(anpr_network STATIC
    src/TcpNetworkTransport.cpp
    src/MessageSchema.cpp
    src/NetworkTransportFactory.cpp
)

target_include_directories(anpr_network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(anpr_network
    PUBLIC
        anpr_core
    PRIVATE
        nlohmann_json::nlohmann_json
)

if (WIN32)
    target_link_libraries(anpr_network PRIVATE ws2_32)
endif()
