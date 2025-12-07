# Proyecto de Aceleración de Inferencia en Edge (NVIDIA Jetson)
![Platform](https://img.shields.io/badge/Platform-NVIDIA_Jetson_Orin_NX-green) ![Language](https://img.shields.io/badge/Language-C++_17_%7C_CUDA-blue) ![Tools](https://img.shields.io/badge/Scripting-Python_3.x-yellow) ![Framework](https://img.shields.io/badge/AI-TensorRT_10_%7C_OpenCV-orange)

## 🚧 Estado del Proyecto: Fase Inicial (Optimización de Hardware)

**Nota Importante:** Este repositorio representa la **primera fase de un proyecto a largo plazo**.

Actualmente, el objetivo principal **no es la precisión (accuracy) del modelo**, sino la **velocidad de inferencia y la optimización del pipeline** en hardware embebido (NVIDIA Jetson). Es posible que notes que el modelo comete errores en la detección o clasificación; esto es esperado, ya que el entrenamiento exhaustivo se realizará en una etapa posterior. Ahora mismo, nuestra prioridad es reducir la latencia, maximizar los FPS y asegurar que la integración con CUDA y TensorRT funcione correctamente.

## Descripción General

Este proyecto implementa un sistema de visión por computadora optimizado para dispositivos de borde (Edge AI) para la detección de células. Hemos migrado la lógica de inferencia a C++ y CUDA para aprovechar al máximo el hardware de la Jetson.

### ¿Por qué decidimos hacerlo así?

1. **C++ & CUDA Nativo:** A diferencia de una implementación pura en Python, aquí utilizamos C++ para la lógica principal y kernels de CUDA (`preprocess.cu`) para el pre-procesamiento de imágenes. Esto libera al CPU de tareas pesadas como la normalización y redimensionamiento.
    
2. **TensorRT:** El modelo ONNX se convierte a un motor serializado (`.engine`) optimizado específicamente para la GPU de la Jetson, permitiendo fusión de capas y precisión FP16.
    
3. **Separación de Responsabilidades:** Tenemos ejecutables dedicados para benchmarking, demostración visual y ejecución en CPU para benchmarking.
    

## Estructura del Proyecto

La organización actual del repositorio es la siguiente:

```
.
├── build/                  # Archivos compilados y artefactos
│   ├── cancer_benchmark    # Ejecutable para pruebas de rendimiento
│   ├── cancer_demo         # Ejecutable para demostración visual
│   ├── cancer_hpc          # Ejecutable principal optimizado
│   └── cancer_detector.engine # Motor TensorRT generado (runtime)
├── CMakeLists.txt          # Configuración de compilación CMake
├── data/                   # Datos de prueba
│   ├── pruebas/            # Imágenes BMP para validación
│   └── test_cell.jpg       # Imagen de prueba unitaria
├── include/                # Cabeceras
│   └── preprocess.h        # Definición de funciones de preprocesamiento
├── models/                 # Modelos de red neuronal
│   ├── cancer_detector_norm.onnx      # Modelo exportado
│   └── cancer_detector_norm.onnx.data # Pesos del modelo ONNX
├── README.MD               # Este archivo
├── scripts/                # Scripts auxiliares
│   └── benchmark_cpu.py    # Comparativa de rendimiento en CPU (Python)
└── src/                    # Código fuente
    ├── main.cpp            # Lógica para 'cancer_hpc'
    ├── main_benchmark.cpp  # Lógica para 'cancer_benchmark'
    ├── main_demo.cpp       # Lógica para 'cancer_demo'
    └── preprocess.cu       # Kernels CUDA para pre-procesamiento de imágenes
```

### Detalle de los componentes clave

- **`src/preprocess.cu`**: Contiene kernels CUDA personalizados. En lugar de usar OpenCV en CPU para redimensionar y normalizar las imágenes, lo hacemos directamente en la GPU, reduciendo la latencia de transferencia de memoria.
    
    
- **`src/main_demo.cpp` (cancer_demo)**: Incluye visualización con OpenCV (`imshow`) para dibujar las cajas delimitadoras (bounding boxes) sobre las detecciones.
    

## Requisitos Previos

- **Hardware:** NVIDIA Jetson (Nano, TX2, Xavier, Orin).
    
- **Sistema Operativo:** JetPack 5.x o 6.x (Ubuntu 20.04/22.04).
    
- **Librerías:**
    
    - CUDA Toolkit
        
    - TensorRT
        
    - OpenCV (con soporte CUDA recomendado)
        
    - CMake
        

## Compilación e Instalación

1. **Crear directorio de construcción:**
    
    ```
    mkdir build
    cd build
    ```
    
2. **Configurar y Compilar:**
    
    ```
    cmake ..
    make -j$(nproc)
    ```
    

Al finalizar, tendrás los ejecutables `cancer_hpc`, `cancer_demo` y `cancer_benchmark` dentro de la carpeta `build/`.

## Ejecución

### 1. Generación del Motor (Engine)

El código espera leer un archivo `.engine`. Si aún no lo tienes generado en `build/`, debes crearlo a partir del ONNX usando `trtexec`:

```
# Ejecutar desde la carpeta build/
/usr/src/tensorrt/bin/trtexec \
  --onnx=../models/cancer_detector_norm.onnx \
  --saveEngine=cancer_detector.engine \
  --fp16
```

### 2. Modos de Uso

A. Modo Demo (Visualización):

Para ver las detecciones en pantalla.

```
./cancer_demo
```

Esto abrirá una ventana mostrando las células detectadas sobre las imágenes de `data/` o el video input configurado.

B. Benchmarking (Pruebas de Estrés):

Para medir latencia y FPS.

```
./cancer_benchmark
```

Adicionalmente, puedes comparar contra la implementación de CPU usando el script de Python:

```
python3 ../scripts/benchmark_cpu.py
```

---

##  Roadmap

1. **Fase 1 (Actual):** Aceleración de hardware con CUDA/TensorRT en Jetson.
    
2. **Fase 2:** Re-entrenamiento con dataset ampliado para mejorar precisión.
    
3. **Fase 3:** Optimización INT8 y calibración.
    
4. **Fase 4:** Despliegue en entorno clínico.
    
---
**Autores:** Alison Celeste Morales Jaramillo, Alonso Delfino Cervantes Flores, Nadia Paulina Ramírez Cristóbal
**Fecha:** Diciembre 2025
**Licencia:** MIT
*Proyecto desarrollado para la clase de High Performance Computing en sistemas embebidos de NVIDIA (2025-1), impartido por **D.Sc. José Eduardo Chairez Veloz** *
