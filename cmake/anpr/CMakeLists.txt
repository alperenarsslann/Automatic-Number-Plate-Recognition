# Top-level CMake project for the layered ANPR embedded system.
#
# Layout:
#   core/        - shared types, config, logging, queues, abstract interfaces
#   capture/     - frame sources (simulated file source, Hikvision RTSP camera)
#   processing/  - plate detection + OCR (IPlateRecognizer implementations)
#   network/     - transport to the site server (INetworkTransport implementations)
#   control/     - application entry point / CLI orchestration
cmake_minimum_required(VERSION 3.16)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(anpr VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV CONFIG REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} found at: ${OpenCV_DIR}")

find_package(nlohmann_json CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)

find_package(Threads REQUIRED)

add_subdirectory(core)
add_subdirectory(capture)
add_subdirectory(processing)
add_subdirectory(network)
add_subdirectory(control)
add_subdirectory(tools/mock_server)

enable_testing()
