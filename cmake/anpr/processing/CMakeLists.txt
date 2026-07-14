# Image processing / ALPR layer: IPlateRecognizer implementations,
# deduplication, factory.
add_library(anpr_processing STATIC
    src/OpenCvDnnPlateRecognizer.cpp
    src/PlateRecognizerFactory.cpp
)

target_include_directories(anpr_processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(anpr_processing
    PUBLIC
        anpr_core
        opencv_dnn
    PRIVATE
        opencv_imgproc
)
