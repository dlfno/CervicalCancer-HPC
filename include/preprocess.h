#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <cstdint> // Necesario para reconocer uint8_t

/**
 * Lanza el kernel de preprocesamiento en la GPU.
 * * @param d_input   Puntero al buffer de imagen de entrada en memoria de dispositivo (GPU).
 * Se espera formato HWC (Height, Width, Channel) entrelazado.
 * @param d_output  Puntero al buffer de salida en memoria de dispositivo (GPU).
 * El resultado será normalizado y en formato CHW (Planar).
 * @param width     Ancho de la imagen.
 * @param height    Alto de la imagen.
 */
void launchPreprocess(uint8_t* d_input, float* d_output, int width, int height);

#endif
