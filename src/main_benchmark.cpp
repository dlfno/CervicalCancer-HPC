#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "NvInfer.h"

using namespace nvinfer1;

// Logger silencioso
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {}
} gLogger;

// Declaración del kernel externo
void launchPreprocess(uint8_t* d_input, float* d_output, int width, int height);

int main() {
    std::cout << "INICIANDO BENCHMARK HPC (NVIDIA JETSON ORIN NX)..." << std::endl;

    // 1. CARGA DEL MOTOR
    std::string engineFile = "cancer_detector.engine"; 
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) { std::cerr << "❌ Error: Falta 'cancer_detector.engine'" << std::endl; return -1; }
    file.seekg(0, file.end); size_t size = file.tellg(); file.seekg(0, file.beg);
    std::vector<char> engineData(size); file.read(engineData.data(), size); file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    IExecutionContext* context = engine->createExecutionContext();

    // 2. PREPARAR DATOS (Imagen Dummy 224x224)
    // Usamos una imagen negra para no depender de archivos externos, el rendimiento es el mismo.
    cv::Mat img = cv::Mat::zeros(224, 224, CV_8UC3);
    
    int width = 224; int height = 224;
    size_t sizeRaw = width * height * 3 * sizeof(uint8_t);
    size_t sizeNet = width * height * 3 * sizeof(float);
    size_t sizeOut = 2 * sizeof(float);

    void *d_in_raw, *d_in_net, *d_out;
    cudaMalloc(&d_in_raw, sizeRaw); cudaMalloc(&d_in_net, sizeNet); cudaMalloc(&d_out, sizeOut);
    cudaStream_t stream; cudaStreamCreate(&stream);

    // 3. WARMUP (Calentamiento)
    // Las primeras ejecuciones en GPU son lentas porque cargan librerías. Las descartamos.
    std::cout << "Calentando GPU (50 iteraciones)..." << std::flush;
    for(int i=0; i<50; ++i) {
        cudaMemcpyAsync(d_in_raw, img.data, sizeRaw, cudaMemcpyHostToDevice, stream);
        launchPreprocess((uint8_t*)d_in_raw, (float*)d_in_net, width, height);
        context->setInputTensorAddress("input", d_in_net);
        context->setOutputTensorAddress("output", d_out);
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
    }
    std::cout << " Listo." << std::endl;

    // 4. BENCHMARK REAL
    int ITERACIONES = 1000;
    std::vector<double> tiempos;
    tiempos.reserve(ITERACIONES);

    std::cout << "⏱️  Ejecutando " << ITERACIONES << " iteraciones de inferencia..." << std::endl;

    for(int i=0; i<ITERACIONES; ++i) {
        // Medimos el ciclo COMPLETO: Host->Device + Preprocess + Inferencia + Device->Host
        // Esto es lo honesto para un sistema real.
        auto start = std::chrono::high_resolution_clock::now();

        cudaMemcpyAsync(d_in_raw, img.data, sizeRaw, cudaMemcpyHostToDevice, stream);
        launchPreprocess((uint8_t*)d_in_raw, (float*)d_in_net, width, height);
        
        context->setInputTensorAddress("input", d_in_net);
        context->setOutputTensorAddress("output", d_out);
        context->enqueueV3(stream);
        
        float logits[2]; // Dummy output
        cudaMemcpyAsync(logits, d_out, sizeOut, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        tiempos.push_back(ms.count());
    }

    // 5. CÁLCULOS ESTADÍSTICOS
    double sum = std::accumulate(tiempos.begin(), tiempos.end(), 0.0);
    double avg = sum / tiempos.size();
    double min_val = *std::min_element(tiempos.begin(), tiempos.end());
    double max_val = *std::max_element(tiempos.begin(), tiempos.end());
    double fps = 1000.0 / avg;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "   RESULTADOS DE RENDIMIENTO (TensorRT)   " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Latencia Mínima:   " << min_val << " ms" << std::endl;
    std::cout << "Latencia Máxima:   " << max_val << " ms" << std::endl;
    std::cout << "Latencia Promedio: " << avg << " ms" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "THROUGHPUT:        " << (int)fps << "QPS" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Limpieza
    cudaStreamDestroy(stream); cudaFree(d_in_raw); cudaFree(d_in_net); cudaFree(d_out);
    delete context; delete engine; delete runtime;
    return 0;
}
