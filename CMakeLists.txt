cmake_minimum_required(VERSION 3.10)
project(CancerDetectorHPC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# 1. Encontrar Paquetes
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Ajusta ruta de TensorRT si es necesario (en Jetson suele detectarse solo)
find_library(NVINFER_LIB nvinfer)

# 2. Incluir directorios
# Le decimos que busque los .h en la carpeta "include"
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

# ==========================================
# EJECUTABLE 1: Versión GPU (HPC)
# ==========================================
# Usamos los archivos dentro de "src/"
add_executable(cancer_hpc 
    src/main.cpp 
    src/preprocess.cu
)

target_link_libraries(cancer_hpc 
    ${OpenCV_LIBS}
    CUDA::cudart
    ${NVINFER_LIB}
)

# ==========================================
# EJECUTABLE 2: Versión CPU (Demo Portátil)
# ==========================================
if(EXISTS "${CMAKE_SOURCE_DIR}/src/main_cpu.cpp")
    add_executable(cancer_cpu 
        src/main_cpu.cpp
    )
    target_link_libraries(cancer_cpu ${OpenCV_LIBS})
endif()

# Arquitecturas CUDA para Jetson (Xavier/Orin/Nano)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 72 87 53) 
endif()
