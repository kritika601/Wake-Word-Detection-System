// tinyml.h
#ifndef TINYML_H
#define TINYML_H
#include "audio_features.h"

#ifdef __cplusplus
extern "C" {
#endif

void tinyml_init(void);
// returns predicted class index (0 = marvin, 1 = unknown), -1 on error
// confidence_out receives the softmax probability of the predicted class
int tinyml_run_inference(float mfcc[MFCC_COEFFS][MFCC_FRAMES], float* confidence_out);

#ifdef __cplusplus
}
#endif
#endif