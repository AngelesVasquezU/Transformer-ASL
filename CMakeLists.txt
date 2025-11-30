cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ViTTransformer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Ruta a LibTorch
# OJO: cámbiala a donde descomprimiste libtorch
set(Torch_DIR "/ruta/a/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(vit_transformer ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(train_vit vit_transformer ${TORCH_LIBRARIES} ${OpenCV_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.cu
)

add_library(vit_transformer STATIC ${SOURCES})
set_property(TARGET vit_transformer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(vit_transformer ${TORCH_LIBRARIES})

# Ejemplo de entrenamiento
add_executable(train_vit examples/train_vit.cpp)
target_link_libraries(train_vit vit_transformer ${TORCH_LIBRARIES})

# Para que encuentre las .so de libtorch al ejecutar
set_property(TARGET vit_transformer PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_property(TARGET train_vit PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
