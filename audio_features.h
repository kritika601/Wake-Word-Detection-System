#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H
#include <stdint.h>

#define MFCC_COEFFS 13
#define MFCC_FRAMES 32


#ifdef __cplusplus
extern "C" {
#endif

void compute_mfcc(const int16_t* audio_in, int num_samples, float mfcc_out[MFCC_COEFFS][MFCC_FRAMES]);



// simple time-domain cleanup: DC removal + high-pass (removes hum/vibration) +
// low-pass (removes hiss) + light AGC (normalizes level). Operates in-place.
void clean_audio_simple(int16_t* samples, int num_samples);

#ifdef __cplusplus
}
#endif
#endif