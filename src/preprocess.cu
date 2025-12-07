#include "preprocess.h"
#include <cuda_runtime.h>

// Kernel para preprocesamiento paralelo en GPU
__global__ void preprocessKernel(uint8_t* src, float* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Índice en la imagen original (OpenCV usa HWC - Height, Width, Channels)
    // Formato BGR
    int idx_src = (y * width + x) * 3;

    // Índice en el tensor de destino (TensorRT usa CHW - Channels, Height, Width)
    // El offset de un plano completo es width * height
    int area = width * height;
    int idx_dst_R = y * width + x;            // Plano 0 (R)
    int idx_dst_G = area + (y * width + x);   // Plano 1 (G)
    int idx_dst_B = 2 * area + (y * width + x); // Plano 2 (B)

    // Normalización: Pixel / 255.0f
    // Conversión BGR (src) a RGB (dst)
    // src[0] es Blue, src[1] es Green, src[2] es Red
    
    dst[idx_dst_R] = (float)src[idx_src + 2] / 255.0f; 
    dst[idx_dst_G] = (float)src[idx_src + 1] / 255.0f;
    dst[idx_dst_B] = (float)src[idx_src + 0] / 255.0f;

    // NOTA: Si tu modelo ResNet fue entrenado con normalización ImageNet (mean/std),
    // deberías restar la media y dividir por la desviación estándar aquí.
    // Ejemplo para ImageNet:
    // dst[idx_dst_R] = (dst[idx_dst_R] - 0.485f) / 0.229f;
}

void launchPreprocess(uint8_t* d_input, float* d_output, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    preprocessKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
}
