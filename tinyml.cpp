// tinyml.cpp
#include <stdio.h>
#include <cmath>
#include <cstdlib>

#include "model_data.h"
#include "tinyml.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"

// !!! REPLACE THESE with the mean/std printed from your training notebook !!!
static const float TRAIN_MEAN = -19.326817;  // <-- fill in
static const float TRAIN_STD  = 107.2952;  // <-- fill in

namespace {
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// static = persists for program lifetime, fixes the dangling-interpreter bug
const tflite::Model* g_tflite_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
}

extern "C" void tinyml_init(void)
{
    g_tflite_model = tflite::GetModel(g_model);
    if (g_tflite_model == nullptr) { printf("Model NULL\r\n"); return; }
    printf("Model OK\r\n");

    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        g_tflite_model, resolver, tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\r\n");
        g_interpreter = nullptr;
        return;
    }

    printf("Interpreter created\r\n");

    TfLiteTensor* input = g_interpreter->input(0);
    printf("input dims: ");
    for (int i = 0; i < input->dims->size; i++) {
        printf("%d ", input->dims->data[i]);
    }
    printf("\r\ninput type: %d, scale: %f, zero_point: %d\r\n",
           input->type, input->params.scale, input->params.zero_point);

    TfLiteTensor* output = g_interpreter->output(0);
    printf("output dims: ");
    for (int i = 0; i < output->dims->size; i++) {
        printf("%d ", output->dims->data[i]);
    }
    printf("\r\noutput type: %d, scale: %f, zero_point: %d\r\n",
           output->type, output->params.scale, output->params.zero_point);
}


extern "C" int tinyml_run_inference(float mfcc[MFCC_COEFFS][MFCC_FRAMES], float* confidence_out)
{
    if (g_interpreter == nullptr) return -1;

    TfLiteTensor* input = g_interpreter->input(0);
    float in_scale = input->params.scale;
    int   in_zero  = input->params.zero_point;

    // normalize with training mean/std, then quantize to int8 using the
    // model's ACTUAL quantization params (read at runtime, not hardcoded)
    int idx = 0;
    for (int c = 0; c < MFCC_COEFFS; c++) {
        for (int f = 0; f < MFCC_FRAMES; f++) {
            float normalized = (mfcc[c][f] - TRAIN_MEAN) / TRAIN_STD;
            int32_t q = (int32_t)lroundf(normalized / in_scale + in_zero);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            input->data.int8[idx++] = (int8_t)q;
        }
    }

    if (g_interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke failed\r\n");
        return -1;
    }

    TfLiteTensor* output = g_interpreter->output(0);
    float out_scale = output->params.scale;
    int   out_zero  = output->params.zero_point;

    float probs[3];
    for (int i = 0; i < 3; i++) {
        probs[i] = (output->data.int8[i] - out_zero) * out_scale;
    }

    int predicted = 0;
    for (int i = 1; i < 3; i++) {
        if (probs[i] > probs[predicted]) predicted = i;
    }
    if (confidence_out) *confidence_out = probs[predicted];
    return predicted;
}