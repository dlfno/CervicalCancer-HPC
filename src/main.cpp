#include "preprocess.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <thread> // Para efecto de pausa
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "NvInfer.h"

using namespace nvinfer1;

// --- COLORES ANSI PARA LA TERMINAL ---
#define RESET   "\033[0m"
#define RED     "\033[1;31m"
#define GREEN   "\033[1;32m"
#define YELLOW  "\033[1;33m"
#define BLUE    "\033[1;34m"
#define CYAN    "\033[1;36m"
#define BOLD    "\033[1m"

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Silencio para la demo
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << RED << "[TRT ERROR] " << msg << RESET << std::endl;
        }
    }
} gLogger;

void launchPreprocess(uint8_t* d_input, float* d_output, int width, int height);

// Función visual de barra de carga
void printProgressBar(int percent) {
    std::cout << "\r" << BLUE << "[Diagnóstico en Curso] " << RESET << "[";
    int pos = percent / 5;
    for (int i = 0; i < 20; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << percent << "%" << std::flush;
}

void applySoftmax(float* input, float* output, int size) {
    float max_val = input[0];
    for(int i = 1; i < size; ++i) if(input[i] > max_val) max_val = input[i];
    float sum = 0.0f;
    for(int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for(int i = 0; i < size; ++i) output[i] /= sum;
}

int main() {
    // --- EFECTO DE INICIO ---
    std::cout << "\033[2J\033[1;1H"; // Limpiar pantalla
    std::cout << CYAN << "================================================" << RESET << std::endl;
    std::cout << CYAN << "   SISTEMA DE DETECCIÓN ONCOLÓGICA - JETSON HPC " << RESET << std::endl;
    std::cout << CYAN << "        Versión 2.5 (Modelo ResNet18-Norm)      " << RESET << std::endl;
    std::cout << CYAN << "================================================" << RESET << std::endl;
    std::cout << std::endl;

    std::cout << YELLOW << "[INFO] Inicializando Motor TensorRT..." << RESET << std::flush;
    
    // 1. CARGA DEL MOTOR
    // El engine suele estar en la misma carpeta del ejecutable (build/)
    std::string engineFile = "cancer_detector.engine"; 
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) { 
        std::cout << RED << " ❌ ERROR (Falta Engine: " << engineFile << ")" << RESET << std::endl; 
        std::cout << YELLOW << "[TIP] Asegúrate de copiar tu .engine a la carpeta 'build/'." << RESET << std::endl;
        return -1; 
    }
    file.seekg(0, file.end); size_t size = file.tellg(); file.seekg(0, file.beg);
    std::vector<char> engineData(size); file.read(engineData.data(), size); file.close();
    
    std::cout << GREEN << "  LISTO" << RESET << std::endl;

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    IExecutionContext* context = engine->createExecutionContext();

    // 2. CARGA DE IMAGEN (CORREGIDO PARA NUEVA ESTRUCTURA)
    // Buscamos en ../data/ porque el ejecutable corre en build/
    std::string imagePath = "../data/test_cell.jpg";
    cv::Mat img = cv::imread(imagePath);
    
    if (img.empty()) { 
        // Intento de fallback: buscar en la carpeta actual por si acaso
        img = cv::imread("test_cell.jpg");
    }

    if (img.empty()) { 
        img = cv::Mat::zeros(224, 224, CV_8UC3); 
        std::cout << RED << "[ALERTA] No se encontró imagen en " << imagePath << ". Usando dummy." << RESET << std::endl;
    } else { 
        cv::resize(img, img, cv::Size(224, 224)); 
    }

    // 3. MEMORIA CUDA
    int width = 224; int height = 224;
    size_t sizeRaw = width * height * 3 * sizeof(uint8_t);
    size_t sizeNet = width * height * 3 * sizeof(float);
    size_t sizeOut = 2 * sizeof(float);

    void *d_in_raw, *d_in_net, *d_out;
    cudaMalloc(&d_in_raw, sizeRaw); cudaMalloc(&d_in_net, sizeNet); cudaMalloc(&d_out, sizeOut);
    cudaStream_t stream; cudaStreamCreate(&stream);

    // --- EFECTO DE PROCESAMIENTO ---
    for(int i=0; i<=100; i+=10) {
        printProgressBar(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Pequeña pausa visual
    }
    std::cout << std::endl << std::endl;

    // 4. INFERENCIA REAL
    cudaMemcpyAsync(d_in_raw, img.data, sizeRaw, cudaMemcpyHostToDevice, stream);
    launchPreprocess((uint8_t*)d_in_raw, (float*)d_in_net, width, height);
    context->setTensorAddress("input", d_in_net);
    context->setTensorAddress("output", d_out);
    context->enqueueV3(stream);
    float logits[2];
    cudaMemcpyAsync(logits, d_out, sizeOut, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 5. LÓGICA MATEMÁTICA
    float probs[2]; 
    applySoftmax(logits, probs, 2); 
    float pRiesgo = probs[1] * 100.0f; // Índice 1 es Riesgo
    float UMBRAL_CORTE = 1.5f;         // Umbral Validado

    // Escalado Visual (Lupa)
    float score_visual = 0.0f;
    if (pRiesgo < UMBRAL_CORTE) {
        score_visual = (pRiesgo / UMBRAL_CORTE) * 45.0f; 
    } else {
        float techo = 20.0f; 
        if (pRiesgo > techo) pRiesgo = techo;
        score_visual = 50.0f + ((pRiesgo - UMBRAL_CORTE) / (techo - UMBRAL_CORTE)) * 50.0f; 
    }

    // --- REPORTE FINAL ---
    std::cout << BOLD << "┌──────────────────────────────────────────────┐" << RESET << std::endl;
    std::cout << BOLD << "│                REPORTE FINAL                 │" << RESET << std::endl;
    std::cout << BOLD << "├──────────────────────────────────────────────┤" << RESET << std::endl;
    
    // Probabilidad Técnica
    std::cout << "│ Probabilidad Neuronal (Raw):      ";
    if (score_visual > 50) std::cout << RED; else std::cout << GREEN;
    std::cout << std::fixed << std::setprecision(2) << probs[1]*100.0f << "%" << RESET;
    if (probs[1]*100.0f < 10.0f) std::cout << "       │" << std::endl;
    else std::cout << "      │" << std::endl;

    // Score de Riesgo Visual
    std::cout << "│ Nivel de Riesgo (Ajustado):       ";
    if (score_visual > 50) std::cout << RED; else std::cout << GREEN;
    std::cout << (int)score_visual << "/100" << RESET << "        │" << std::endl;

    std::cout << BOLD << "├──────────────────────────────────────────────┤" << RESET << std::endl;

    if (score_visual >= 50.0f) {
        std::cout << "│ DIAGNÓSTICO:    " << RED << " ANOMALÍA DETECTADA " << RESET << "      │" << std::endl;
        std::cout << "│ CLASIFICACIÓN:  " << RED << "    POSIBLE CARCINOMA    " << RESET << "      │" << std::endl;
    } else {
        std::cout << "│ DIAGNÓSTICO:    " << GREEN << " CÉLULA SANA            " << RESET << "      │" << std::endl;
        std::cout << "│ CLASIFICACIÓN:  " << GREEN << "    TEJIDO NORMAL        " << RESET << "      │" << std::endl;
    }
    std::cout << BOLD << "└──────────────────────────────────────────────┘" << RESET << std::endl;
    std::cout << std::endl;

    // Limpieza
    cudaStreamDestroy(stream); cudaFree(d_in_raw); cudaFree(d_in_net); cudaFree(d_out);
    delete context; delete engine; delete runtime;
    return 0;
}
