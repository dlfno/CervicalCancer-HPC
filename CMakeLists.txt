cmake_minimum_required(VERSION 3.10)
project(CervicalCancerHPC LANGUAGES CXX CUDA)

# Estándares modernos para C++ y CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 1. Encontrar Dependencias
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Buscar TensorRT (Librería nvinfer)
# En Jetson suele estar en rutas estándar, esto ayuda a encontrarla
find_library(NVINFER_LIB nvinfer REQUIRED)

# 2. Incluir directorios de cabeceras (headers)
# 'include' es para preprocess.h
include_directories(include) 
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# =========================================================
# 3. Definición de Ejecutables
# =========================================================

# --- A. CANCER DEMO (Visualización) ---
add_executable(cancer_demo 
    src/main_demo.cpp 
    src/preprocess.cu
)

# Enlazar librerías necesarias
target_link_libraries(cancer_demo 
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIB}
)

# --- B. CANCER BENCHMARK (Rendimiento) ---
add_executable(cancer_benchmark 
    src/main_benchmark.cpp 
    src/preprocess.cu
)

# Enlazar librerías necesarias
target_link_libraries(cancer_benchmark 
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIB}
)


