#include "audio_features.h"
#include "mel_filterbank.h"
#include "dct_matrix.h"
#include "third_party/third_party/kissfft/tools/kiss_fftr.h"
#include "compiler/compiler_ld.h" 
#include <cmath>
#include <cstring>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define N_FFT 2048
#define HOP_LENGTH 512
#define N_MELS 128

static float hann_window[N_FFT];
static bool hann_init = false;

static void init_hann() {
    for (int i = 0; i < N_FFT; i++) {
        hann_window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (N_FFT - 1));
    }
    hann_init = true;
}

/* ---- shared scratch buffers, used by ALL of compute_mfcc / compute_noise_profile / denoise_audio ----
   ONE copy each, in PSRAM, instead of each function keeping its own static copy in fast RAM */
static ATTR_NOINIT_PSRAM_SECTION uint8_t g_kiss_mem[32 * 1024];
static ATTR_NOINIT_PSRAM_SECTION float g_windowed[N_FFT];
static ATTR_NOINIT_PSRAM_SECTION kiss_fft_cpx g_fft_out[N_FFT / 2 + 1];
static ATTR_NOINIT_PSRAM_SECTION float g_padded_in[16000 + N_FFT];

// static uint8_t g_kiss_mem[32 * 1024];
// static float g_windowed[N_FFT];
// static kiss_fft_cpx g_fft_out[N_FFT / 2 + 1];
// static float g_padded_in[16000 + N_FFT];

//temp test
static ATTR_NOINIT_PSRAM_SECTION float g_mel_filterbank_psram[N_MELS * (N_FFT / 2 + 1)];
static ATTR_NOINIT_PSRAM_SECTION float g_dct_matrix_psram[MFCC_COEFFS * N_MELS];
static bool g_tables_copied = false;

//temp test
static void ensure_tables_in_psram()
{
    if (g_tables_copied) return;
    memcpy(g_mel_filterbank_psram, mel_filterbank, sizeof(g_mel_filterbank_psram));
    memcpy(g_dct_matrix_psram, dct_matrix, sizeof(g_dct_matrix_psram));
    g_tables_copied = true;
}


static kiss_fftr_cfg get_fft_cfg()
{
    size_t kiss_mem_len = sizeof(g_kiss_mem);
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, g_kiss_mem, &kiss_mem_len);
    if (cfg == nullptr) {
        printf("kiss_fftr_alloc FAILED\r\n");
    }
    return cfg;
}

void compute_mfcc(const int16_t* audio_in, int num_samples, float mfcc_out[MFCC_COEFFS][MFCC_FRAMES]) {
    ensure_tables_in_psram();
    if (!hann_init) init_hann();

    int pad = N_FFT / 2;
    int padded_len = num_samples + 2 * pad;
    memset(g_padded_in, 0, sizeof(g_padded_in));
    for (int i = 0; i < num_samples; i++) {
        g_padded_in[pad + i] = audio_in[i] / 32768.0f;
    }

    kiss_fftr_cfg fft_cfg = get_fft_cfg();
    if (fft_cfg == nullptr) return;

    static float power_spec[N_FFT / 2 + 1];
    static float mel_energy[N_MELS];


    for (int frame = 0; frame < MFCC_FRAMES; frame++) {
        int start = frame * HOP_LENGTH;
        if (start + N_FFT > padded_len) break;

        for (int i = 0; i < N_FFT; i++) {
            g_windowed[i] = g_padded_in[start + i] * hann_window[i];
        }

        kiss_fftr(fft_cfg, g_windowed, g_fft_out);

        for (int i = 0; i <= N_FFT / 2; i++) {
            float re = g_fft_out[i].r, im = g_fft_out[i].i;
            power_spec[i] = re * re + im * im;
        }

        for (int m = 0; m < N_MELS; m++) {
            float sum = 0.0f;
            // const float* row = &mel_filterbank[m * (N_FFT / 2 + 1)];
            const float* row = &g_mel_filterbank_psram[m * (N_FFT / 2 + 1)];  
            for (int i = 0; i <= N_FFT / 2; i++) {
                sum += power_spec[i] * row[i];
            }
            mel_energy[m] = 10.0f * log10f(fmaxf(sum, 1e-10f));
        }

        for (int c = 0; c < MFCC_COEFFS; c++) {
            float sum = 0.0f;
            // const float* row = &dct_matrix[c * N_MELS];
            const float* row = &g_dct_matrix_psram[c * N_MELS];
            for (int m = 0; m < N_MELS; m++) {
                sum += mel_energy[m] * row[m];
            }
            mfcc_out[c][frame] = sum;
        }
    }

   
}

//noise reduction

static void remove_dc_offset(int16_t* samples, int N)
{
    int64_t sum = 0;
    for (int i = 0; i < N; i++) sum += samples[i];
    int16_t dc = (int16_t)(sum / N);
    for (int i = 0; i < N; i++) {
        int32_t v = samples[i] - dc;
        samples[i] = (int16_t)v;
    }
}

// one-pole high-pass, ~100Hz cutoff @ 16kHz -- removes hum/vibration, keeps speech
static void highpass_filter(int16_t* samples, int N)
{
    const float alpha = 0.96f;
    float y = 0.0f;
    float prev_x = 0.0f;
    for (int i = 0; i < N; i++) {
        float x = (float)samples[i];
        y = alpha * (y + x - prev_x);
        prev_x = x;
        float clamped = y;
        if (clamped > 32767.0f) clamped = 32767.0f;
        if (clamped < -32768.0f) clamped = -32768.0f;
        samples[i] = (int16_t)clamped;
    }
}

// one-pole low-pass, ~4kHz cutoff @ 16kHz -- removes high-frequency hiss
static void lowpass_filter(int16_t* samples, int N)
{
    const float beta = 0.6f;
    float y = (float)samples[0];
    for (int i = 0; i < N; i++) {
        float x = (float)samples[i];
        y = y + beta * (x - y);
        samples[i] = (int16_t)y;
    }
}

// light AGC: normalize toward a target RMS, with a capped max gain so pure
// noise doesn't get amplified into nonsense
static void apply_agc(int16_t* samples, int N)
{
    const float target_rms = 3000.0f;
    const float max_gain   = 8.0f;

    int64_t sum_sq = 0;
    for (int i = 0; i < N; i++) {
        int32_t s = samples[i];
        sum_sq += (int64_t)s * s;
    }
    float rms = sqrtf((float)sum_sq / N);
    if (rms < 1.0f) return;  // avoid divide-by-near-zero on pure silence

    float gain = target_rms / rms;
    if (gain > max_gain) gain = max_gain;
    if (gain < 1.0f) gain = 1.0f;  // never attenuate, only boost

    for (int i = 0; i < N; i++) {
        float v = samples[i] * gain;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        samples[i] = (int16_t)v;
    }
}

void clean_audio_simple(int16_t* samples, int num_samples)
{
    remove_dc_offset(samples, num_samples);
    highpass_filter(samples, num_samples);
    lowpass_filter(samples, num_samples);
    apply_agc(samples, num_samples);
}