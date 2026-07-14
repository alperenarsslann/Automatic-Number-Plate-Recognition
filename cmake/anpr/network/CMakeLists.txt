# Network / API layer: INetworkTransport implementations + message schema.
# INTERFACE for now (headers only); becomes STATIC in step 4 when the TCP
# transport and the mock test server are implemented.
add_library(anpr_network INTERFACE)

target_include_directories(anpr_network INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(anpr_network
    INTERFACE
        anpr_core
)
