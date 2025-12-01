cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ViTTransformer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# LibTorch
set(Torch_DIR "/home/daniela/Descargas/libtorch-shared-with-deps-2.9.1+cu130/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

#MOVER ESTE include_directories ANTES DE add_library Y add_executable
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Archivos fuente
file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/*.cu
)

# Biblioteca estática Vit
add_library(vit_transformer STATIC ${SOURCES})
set_property(TARGET vit_transformer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(vit_transformer ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# Ejecutable de entrenamiento
add_executable(train_vit examples/train_vit.cpp)
target_link_libraries(train_vit vit_transformer ${TORCH_LIBRARIES} ${OpenCV_LIBS})

add_executable(infer_vit examples/infer_vit.cpp)
target_link_libraries(infer_vit vit_transformer ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET infer_vit PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Para que encuentre las .so de libtorch al ejecutar
set_property(TARGET vit_transformer PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_property(TARGET train_vit PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")