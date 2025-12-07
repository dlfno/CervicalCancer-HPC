# Aceleración de Redes Neuronales para Detección de Cáncer Cervicouterino

Este proyecto implementa un sistema de detección temprana de células cancerígenas optimizado para sistemas embebidos de alto rendimiento (Edge AI). Utiliza **C++** y **CUDA** para el preprocesamiento de imágenes y **TensorRT** para la inferencia, logrando una aceleración significativa respecto a ejecuciones estándar en CPU.

## 🚀 Características
- **Inferencia Acelerada:** Uso de TensorRT en NVIDIA Jetson Orin NX.
- **Preprocesamiento en GPU:** Kernel CUDA personalizado (`preprocess.cu`) para normalización y redimensionamiento.
- **Pipeline Híbrido:** Validación de modelos en Python y despliegue en C++.
- **Benchmark:** Comparativa de rendimiento entre CPU y GPU.

## 🛠️ Requisitos
- **Hardware:** NVIDIA Jetson (Orin NX recomendado) o GPU con soporte CUDA.
- **Software:**
  - JetPack 5.x / 6.x
  - CUDA 11.4+
  - TensorRT 8.x
  - OpenCV
  - CMake

## 📂 Estructura del Proyecto
- `src/`: Código fuente C++ y Kernels CUDA.
- `models/`: Modelo ONNX entrenado.
- `scripts/`: Utilidades en Python para validación y gráficas.

## ⚙️ Compilación e Instalación

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/tu-usuario/proyecto-cancer-hpc.git](https://github.com/tu-usuario/proyecto-cancer-hpc.git)
   cd proyecto-cancer-hpc
