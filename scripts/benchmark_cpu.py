import time
import torch
import torchvision.models as models

def main():
    print("========================================")
    print("   BENCHMARK CPU (PyTorch nativo)       ")
    print("========================================")
    
    # 1. Cargar la arquitectura ResNet18
    # No necesitamos los pesos específicos para medir la VELOCIDAD, 
    # la estructura matemática es la misma.
    print(">>> Cargando arquitectura ResNet18 en CPU...")
    try:
        model = models.resnet18(weights=None) # Estructura vacía (más rápido de cargar)
    except:
        model = models.resnet18(pretrained=False) # Compatibilidad versiones viejas
        
    model.eval() # Modo evaluación

    # 2. Crear una imagen falsa (Dummy input)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 3. Calentamiento (Warm-up)
    print(">>> Calentando CPU (5 ciclos)...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)

    # 4. Benchmark Real
    iterations = 1000
    print(f">>> Midiendo {iterations} iteraciones...")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(iterations):
            _ = model(input_tensor)
            
    end_time = time.time()

    # 5. Cálculos
    total_time = end_time - start_time
    avg_latency_sec = total_time / iterations
    avg_latency_ms = avg_latency_sec * 1000
    fps = 1.0 / avg_latency_sec

    print("\n---------------- RESULTADOS CPU ----------------")
    print(f" Tiempo Total:      {total_time:.4f} s")
    print(f" Latencia Promedio: {avg_latency_ms:.2f} ms")
    print(f" Throughput:        {fps:.2f} FPS")
    print("------------------------------------------------\n")

if __name__ == "__main__":
    main()
