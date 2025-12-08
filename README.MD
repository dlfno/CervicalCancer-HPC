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

    

## Requisitos Previos

- **Hardware:** NVIDIA Jetson (Nano, TX2, Xavier, Orin).
    
- **Sistema Operativo:** JetPack 5.x o 6.x (Ubuntu 20.04/22.04).
    
- **Librerías:**
    
    - CUDA Toolkit
        
    - TensorRT
        
    - OpenCV (con soporte CUDA recomendado)
        
    - CMake
        
---

## Descripción Técnica de Módulos

El proyecto se compone de cuatro piezas fundamentales que separan la lógica de demostración, validación de rendimiento y optimización de hardware:

### 1. `src/main_demo.cpp` (Validación Funcional)
Es el punto de entrada para verificar la **corrección del sistema**.
* **Función:** Carga el motor TensorRT y procesa una imagen individual (`test_cell.jpg`) para simular un diagnóstico clínico en tiempo real.
* **Salida:** Interfaz visual en terminal (CLI) con barras de carga y reporte de probabilidad (Sano/Anomalía). Sirve para demostrar que el pipeline de inferencia funciona correctamente de principio a fin.

### 2. `src/main_benchmark.cpp` (Análisis de Rendimiento HPC)
Módulo crítico diseñado para estresar el hardware y obtener métricas puras.
* **Función:** Ejecuta un bucle de 1000 iteraciones de inferencia continua sobre la GPU, midiendo el ciclo completo (Host-to-Device transfer + Preprocesamiento + Inferencia + Device-to-Host transfer).
* **Salida:** Calcula y reporta la **Latencia Promedio (ms)** y el **Throughput (FPS)** real del sistema optimizado.

### 3. `src/preprocess.cu` (Aceleración por Hardware)
Contiene la implementación de bajo nivel necesaria para liberar al CPU.
* **Función:** Define `kernels` de CUDA personalizados que se ejecutan directamente en los núcleos de la GPU.
* **Tarea:** Realiza la normalización de píxeles, reordenamiento de canales (HWC a CHW) y escalado de imagen sin mover los datos de la memoria gráfica, eliminando cuellos de botella en el bus de memoria.

### 4. `scripts/benchmark_cpu.py` (Línea Base)
Script de Python utilizado para establecer el punto de comparación (Baseline).
* **Función:** Ejecuta la misma arquitectura neuronal utilizando PyTorch estándar sobre el CPU.
* **Objetivo:** Permite calcular el **Speedup (x veces más rápido)** que ofrece nuestra implementación en C++/CUDA frente a una implementación tradicional en software.

### 5. `include/preprocess.h` (El Puente C++/CUDA)
Es el archivo de cabecera que conecta dos mundos distintos.
* **Función:** Actúa como interfaz o "contrato" entre el código C++ estándar y el compilador de NVIDIA (NVCC).
* **Tarea:** Declara la función `launchPreprocess`, permitiendo que los ejecutables principales (`main_demo` y `main_benchmark`) puedan invocar los kernels de la GPU sin necesidad de conocer la sintaxis compleja de CUDA.

## Compilación e Instalación

Sigue estos pasos estrictamente desde la terminal de tu Jetson:

1.  **Clonar el repositorio y acceder a la carpeta:**
    ```bash
    git clone [https://github.com/dlfno/CervicalCancer-HPC.git](https://github.com/dlfno/CervicalCancer-HPC.git)
    cd CervicalCancer-HPC
    ```

2.  **Crear el directorio de construcción:**
    ```bash
    mkdir -p build
    cd build
    ```

3.  **Compilar el proyecto:**
    ```bash
    cmake ..
    make -j$(nproc)
    ```

> **Nota:** Al finalizar, deberás ver los ejecutables `cancer_demo` y `cancer_benchmark` dentro de la carpeta `build/`.

---

## Guía de Ejecución

Todos los comandos deben ejecutarse desde la carpeta `build/` creada en el paso anterior.

### Paso 1: Generar el Motor TensorRT (Solo la primera vez)
El código C++ requiere un motor optimizado. Genera el archivo `.engine` a partir del modelo ONNX usando `trtexec`:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=../models/cancer_detector_norm.onnx \
  --saveEngine=cancer_detector.engine \
  --fp16
````

_Si este paso no se realiza, los demos fallarán._

### Paso 2: Ejecutar Demo Visual

Muestra el diagnóstico simulado en la terminal procesando una imagen de prueba.

Bash

```
./cancer_demo
```

### Paso 3: Benchmark de Rendimiento (HPC)

Mide la latencia y los FPS reales utilizando la GPU + TensorRT.

Bash

```
./cancer_benchmark
```

### Paso 4: Comparativa con CPU 

Para contrastar la velocidad obtenida, ejecuta el script de Python que corre en CPU pura:

Bash

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

_Proyecto desarrollado para la clase de High Performance Computing en sistemas embebidos de NVIDIA (2025-1), impartido por **D.Sc. José Eduardo Chairez Veloz**_
