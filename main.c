/* ============================================================
   Continuous wake-word detection: ping-pong AUADC capture ->
   rolling history buffer -> energy-windowed MFCC -> TFLM inference
   ============================================================ */

#include "qcc74x_auadc.h"
#include "qcc74x_audac.h"
#include "qcc74x_gpio.h"
#include "qcc74x_dma.h"

#include "qcc743_glb.h"
#include "board.h"
#include <math.h>
#include "tinyml.h"
#include "audio_features.h"

#define AUADC_SAMPLE_RATE_HZ   32000

#define AUADC_LLI_POOL_CNT     150   /* raw buff ~640000 bytes */
#define AUDAC_LLI_POOL_CNT     80    /* pcm buff ~320000 bytes */

#define MFCC_INPUT_SAMPLES 16000  // 1 second @ 16kHz after downsampling

#define CHUNK_SECONDS       0.5
#define CHUNK_SAMPLE_NUM    (uint32_t)(AUADC_SAMPLE_RATE_HZ * CHUNK_SECONDS)   // 16000 samples
#define ROLLING_HISTORY_SAMPLES  32000   // 2 seconds @ 16kHz, same size as your old full-capture buffer

static int16_t audio_16k[MFCC_INPUT_SAMPLES];
static float mfcc_buffer[MFCC_COEFFS][MFCC_FRAMES];

static ATTR_NOINIT_PSRAM_SECTION __attribute((aligned(32))) uint32_t auadc_raw_buff[2][CHUNK_SAMPLE_NUM];
static ATTR_NOINIT_PSRAM_SECTION int16_t rolling_history[ROLLING_HISTORY_SAMPLES];

/* dedicated buffer for playing back just the extracted 1s window that gets fed to the model */
static ATTR_NOINIT_PSRAM_SECTION __attribute((aligned(32))) uint16_t window_playback_buff[MFCC_INPUT_SAMPLES * 2];

static volatile int filling_index = 0;   // which buffer DMA is CURRENTLY writing into
static volatile int ready_index   = -1;  // which buffer just finished and is ready to process (-1 = none pending)
static volatile uint32_t rolling_write_pos = 0;   // next write position in rolling_history (wraps around)
static volatile bool rolling_history_full = false; // becomes true once we've wrapped around once

static struct qcc74x_dma_channel_lli_pool_s auadc_dma_lli_pool[AUADC_LLI_POOL_CNT];
static struct qcc74x_dma_channel_lli_pool_s audac_dma_lli_pool[AUDAC_LLI_POOL_CNT];

static struct qcc74x_device_s *auadc_hd;
static struct qcc74x_device_s *auadc_dma_hd;
static struct qcc74x_device_s *audac_hd;
static struct qcc74x_device_s *audac_dma_hd;

static volatile bool playback_done = false;

typedef enum {
    STATE_LISTENING,
    STATE_TASK_ACTIVE
} system_state_t;

static system_state_t current_state = STATE_LISTENING;

#define DEBUG_PLAYBACK_WINDOW   1   // set to 0 to disable audible playback 

/* ================= GPIO ================= */
static struct qcc74x_device_s *gpio_hd;   // add this near your other device handle globals (auadc_hd, audac_hd, etc.)

static void audio_gpio_init(void)
{
    gpio_hd = qcc74x_device_get_by_name("gpio");

    GLB_Set_PDM_IO_Sel(GLB_PDM_IO_SEL_AUPDM_TOP);
    qcc74x_gpio_init(gpio_hd, GPIO_PIN_25,
        GPIO_FUNC_PDM | GPIO_ALTERNATE | GPIO_FLOAT | GPIO_SMT_EN | GPIO_DRV_2);
    qcc74x_gpio_init(gpio_hd, GPIO_PIN_26,
        GPIO_FUNC_PDM | GPIO_ALTERNATE | GPIO_FLOAT | GPIO_SMT_EN | GPIO_DRV_2);

    qcc74x_gpio_init(gpio_hd, GPIO_PIN_20,
        GPIO_OUTPUT | GPIO_PULLDOWN | GPIO_SMT_EN | GPIO_DRV_0);
    qcc74x_gpio_reset(gpio_hd, GPIO_PIN_20);

    qcc74x_gpio_init(gpio_hd, GPIO_PIN_14,
        GPIO_FUNC_AUDAC_PWM | GPIO_ALTERNATE | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_2);
    qcc74x_gpio_init(gpio_hd, GPIO_PIN_15,
        GPIO_FUNC_AUDAC_PWM | GPIO_ALTERNATE | GPIO_PULLUP | GPIO_SMT_EN | GPIO_DRV_2);

    /* wake-word indicator LED */
    qcc74x_gpio_init(gpio_hd, GPIO_PIN_1,
        GPIO_OUTPUT | GPIO_PULLDOWN | GPIO_SMT_EN | GPIO_DRV_0);
    qcc74x_gpio_reset(gpio_hd, GPIO_PIN_1);   // start OFF
}



/* ================= RECORD (AUADC, 32-bit PDM) ================= */
static void auadc_dma_callback(void *arg)
{
    int finished = filling_index;
    qcc74x_l1c_dcache_invalidate_range(auadc_raw_buff[finished], sizeof(auadc_raw_buff[finished]));

    filling_index = 1 - filling_index;   // swap to the other buffer

    // immediately restart DMA into the new buffer, minimal gap
    struct qcc74x_dma_channel_lli_transfer_s transfer;
    transfer.src_addr = (uint32_t)DMA_ADDR_AUADC_RDR;
    transfer.dst_addr = (uint32_t)auadc_raw_buff[filling_index];
    transfer.nbytes   = sizeof(auadc_raw_buff[filling_index]);

    qcc74x_dma_channel_lli_reload(auadc_dma_hd, auadc_dma_lli_pool, AUADC_LLI_POOL_CNT, &transfer, 1);
    qcc74x_dma_channel_start(auadc_dma_hd);

    ready_index = finished;   // tell main loop this buffer is ready to read
}

static void auadc_init(void)
{
    struct qcc74x_auadc_init_config_s cfg = {
        .sampling_rate  = AUADC_SAMPLING_RATE_32K,
        .input_mode     = AUADC_INPUT_MODE_PDM_L,
        .data_format    = AUADC_DATA_FORMAT_32BIT,
        .fifo_threshold = 3,
    };
    GLB_Config_AUDIO_PLL_To_491P52M();
    GLB_PER_Clock_UnGate(GLB_AHB_CLOCK_AUDIO);
    auadc_hd = qcc74x_device_get_by_name("auadc");
    qcc74x_auadc_init(auadc_hd, &cfg);
    qcc74x_auadc_link_rxdma(auadc_hd, true);
}

static void auadc_dma_init(void)
{
    struct qcc74x_dma_channel_lli_transfer_s transfers[1];
    struct qcc74x_dma_channel_config_s cfg = {
        .direction       = DMA_PERIPH_TO_MEMORY,
        .src_req         = DMA_REQUEST_AUADC_RX,
        .dst_req         = DMA_REQUEST_NONE,
        .src_addr_inc    = DMA_ADDR_INCREMENT_DISABLE,
        .dst_addr_inc    = DMA_ADDR_INCREMENT_ENABLE,
        .src_burst_count = DMA_BURST_INCR4,
        .dst_burst_count = DMA_BURST_INCR4,
        .src_width       = DMA_DATA_WIDTH_32BIT,
        .dst_width       = DMA_DATA_WIDTH_32BIT,
    };
    auadc_dma_hd = qcc74x_device_get_by_name("dma0_ch0");
    qcc74x_dma_channel_init(auadc_dma_hd, &cfg);
    qcc74x_dma_channel_irq_attach(auadc_dma_hd, auadc_dma_callback, NULL);

    filling_index = 0;
    transfers[0].src_addr = (uint32_t)DMA_ADDR_AUADC_RDR;
    transfers[0].dst_addr = (uint32_t)auadc_raw_buff[0];
    transfers[0].nbytes   = sizeof(auadc_raw_buff[0]);

    qcc74x_dma_channel_lli_reload(auadc_dma_hd, auadc_dma_lli_pool, AUADC_LLI_POOL_CNT, transfers, 1);
}

#define SEARCH_WINDOW_SAMPLES  24000   // 1.5 seconds @ 16kHz -- search space to look for the word within

static void extract_best_recent_window(int16_t* out_window)
{
    static int16_t search_buf[SEARCH_WINDOW_SAMPLES];

    // pull the last 1.5s out of the rolling buffer 
    uint32_t end = rolling_write_pos;
    uint32_t start;

    if (end >= SEARCH_WINDOW_SAMPLES) {
        start = end - SEARCH_WINDOW_SAMPLES;
        memcpy(search_buf, &rolling_history[start], SEARCH_WINDOW_SAMPLES * sizeof(int16_t));
    } else {
        uint32_t tail_len = SEARCH_WINDOW_SAMPLES - end;
        start = ROLLING_HISTORY_SAMPLES - tail_len;
        memcpy(search_buf, &rolling_history[start], tail_len * sizeof(int16_t));
        memcpy(search_buf + tail_len, &rolling_history[0], end * sizeof(int16_t));
    }

    // slide a 1-second window across this 1.5s search space, pick the highest-energy slice
    int64_t best_energy = -1;
    int best_start = 0;
    int step = 400;   // check every 25ms

    for (int s = 0; s + MFCC_INPUT_SAMPLES <= SEARCH_WINDOW_SAMPLES; s += step) {
        int64_t sum_sq = 0;
        for (int i = 0; i < MFCC_INPUT_SAMPLES; i++) {
            int32_t v = search_buf[s + i];
            sum_sq += (int64_t)v * v;
        }
        if (sum_sq > best_energy) {
            best_energy = sum_sq;
            best_start = s;
        }
    }

    memcpy(out_window, &search_buf[best_start], MFCC_INPUT_SAMPLES * sizeof(int16_t));
}

/* ================= PLAYBACK (AUDAC, 16-bit PCM) ================= */
static void audac_dma_callback(void *arg)
{
    playback_done = true;
}

static void audac_init(void)
{
    struct qcc74x_audac_init_config_s cfg = {
        .sampling_rate       = AUDAC_SAMPLING_RATE_32K,
        .output_mode         = AUDAC_OUTPUT_MODE_PWM,
        .source_channels_num = AUDAC_SOURCE_CHANNEL_SINGLE,
        .mixer_mode          = AUDAC_MIXER_MODE_ONLY_L,
        .data_format         = AUDAC_DATA_FORMAT_16BIT,
        .fifo_threshold      = 3,
    };
    struct qcc74x_audac_volume_config_s vcfg = {
        .mute_ramp_en              = true,
        .mute_up_ramp_rate         = AUDAC_RAMP_RATE_FS_32,
        .mute_down_ramp_rate       = AUDAC_RAMP_RATE_FS_8,
        .volume_update_mode        = AUDAC_VOLUME_UPDATE_MODE_RAMP,
        .volume_ramp_rate          = AUDAC_RAMP_RATE_FS_128,
        .volume_zero_cross_timeout = AUDAC_RAMP_RATE_FS_128,
    };
    GLB_Config_AUDIO_PLL_To_491P52M();
    GLB_PER_Clock_UnGate(GLB_AHB_CLOCK_AUDIO);
    audac_hd = qcc74x_device_get_by_name("audac");
    qcc74x_audac_init(audac_hd, &cfg);
    qcc74x_audac_feature_control(audac_hd, AUDAC_CMD_SET_VOLUME_VAL, (size_t)(-5 * 2));
    qcc74x_audac_volume_init(audac_hd, &vcfg);
    qcc74x_audac_link_rxdma(audac_hd, true);
}

/* naive 16kHz -> 32kHz upsample by sample duplication, just for audible playback/debugging */
static void upsample_16k_to_32k_for_playback(void)
{
    for (uint32_t i = 0; i < MFCC_INPUT_SAMPLES; i++) {
        window_playback_buff[i * 2]     = (uint16_t)audio_16k[i];
        window_playback_buff[i * 2 + 1] = (uint16_t)audio_16k[i];
    }
}

static void play_window_buffer(void)
{
    struct qcc74x_dma_channel_lli_transfer_s transfers[1];
    struct qcc74x_dma_channel_config_s cfg = {
        .direction       = DMA_MEMORY_TO_PERIPH,
        .src_req         = DMA_REQUEST_NONE,
        .dst_req         = DMA_REQUEST_AUDAC_TX,
        .src_addr_inc    = DMA_ADDR_INCREMENT_ENABLE,
        .dst_addr_inc    = DMA_ADDR_INCREMENT_DISABLE,
        .src_burst_count = DMA_BURST_INCR4,
        .dst_burst_count = DMA_BURST_INCR4,
        .src_width       = DMA_DATA_WIDTH_16BIT,
        .dst_width       = DMA_DATA_WIDTH_16BIT,
    };

    playback_done = false;

    audac_init();

    audac_dma_hd = qcc74x_device_get_by_name("dma0_ch1");
    qcc74x_dma_channel_init(audac_dma_hd, &cfg);
    qcc74x_dma_channel_irq_attach(audac_dma_hd, audac_dma_callback, NULL);

    transfers[0].src_addr = (uint32_t)window_playback_buff;
    transfers[0].dst_addr = (uint32_t)DMA_ADDR_AUDAC_TDR;
    transfers[0].nbytes   = sizeof(window_playback_buff);

    qcc74x_l1c_dcache_clean_range(window_playback_buff, sizeof(window_playback_buff));

    qcc74x_dma_channel_lli_reload(audac_dma_hd, audac_dma_lli_pool, AUDAC_LLI_POOL_CNT, transfers, 1);
    qcc74x_dma_channel_start(audac_dma_hd);
    qcc74x_audac_feature_control(audac_hd, AUDAC_CMD_PLAY_START, 0);
    qcc74x_audac_feature_control(audac_hd, AUDAC_CMD_SET_VOLUME_VAL, 2);
    printf("playing back the extracted window...\r\n");
    while (!playback_done) {
        qcc74x_mtimer_delay_ms(50);
    }
    printf("window playback done\r\n");
}

static void append_chunk_to_rolling_history(const uint32_t* raw_chunk)
{
    // raw_chunk is CHUNK_SAMPLE_NUM (16000) samples @ 32kHz, 32-bit PDM format --
    // same conversion as convert_raw_to_pcm() + downsample, just done per-chunk now
    const uint32_t chunk_16k_samples = CHUNK_SAMPLE_NUM / 2;   // 8000 samples after 2:1 downsample

    for (uint32_t i = 0; i < chunk_16k_samples; i++) {
        // convert two 32-bit PDM raw samples -> two 16-bit PCM samples -> average down to one 16kHz sample
        int32_t raw_a = (int32_t)raw_chunk[i * 2];
        int32_t raw_b = (int32_t)raw_chunk[i * 2 + 1];

        int32_t shifted_a = raw_a >> 16;
        int32_t shifted_b = raw_b >> 16;
        if (shifted_a > 32767) shifted_a = 32767;
        if (shifted_a < -32768) shifted_a = -32768;
        if (shifted_b > 32767) shifted_b = 32767;
        if (shifted_b < -32768) shifted_b = -32768;

        int16_t sample_16k = (int16_t)((shifted_a + shifted_b) / 2);

        rolling_history[rolling_write_pos] = sample_16k;
        rolling_write_pos++;
        if (rolling_write_pos >= ROLLING_HISTORY_SAMPLES) {
            rolling_write_pos = 0;
            rolling_history_full = true;
        }
    }
}

static void perform_wake_word_task(void)
{
    printf(">>> TASK MODE: wake word detected, LED ON <<<\r\n");
    qcc74x_gpio_set(gpio_hd, GPIO_PIN_1);   // turn LED on

    // TODO: replace with your real task logic later (spoken response, etc.)
    qcc74x_mtimer_delay_ms(3000);   // LED stays on for 3s 

    qcc74x_gpio_reset(gpio_hd, GPIO_PIN_1);   // turn LED off
    printf(">>> TASK COMPLETE: LED OFF, returning to listening <<<\r\n");
}

int main(void)
{
    board_init();
    tinyml_init();

    printf("ping-pong capture test: start\r\n");
    audio_gpio_init();
    auadc_init();
    auadc_dma_init();

    qcc74x_dma_channel_start(auadc_dma_hd);
    qcc74x_auadc_feature_control(auadc_hd, AUADC_CMD_RECORD_START, 0);

    uint32_t chunk_count = 0;

    #define DETECTION_INTERVAL_MS   400   // run inference this often; tune later

    while (1) {
        if (ready_index != -1) {
            int idx = ready_index;
            ready_index = -1;
            append_chunk_to_rolling_history(auadc_raw_buff[idx]);
            chunk_count++;
        }

        static uint32_t last_detection_tick = 0;
        static uint32_t tick_ms = 0;
        tick_ms += 5;

        if (current_state == STATE_LISTENING) {
            if (rolling_history_full && (tick_ms - last_detection_tick >= DETECTION_INTERVAL_MS)) {
                last_detection_tick = tick_ms;

                extract_best_recent_window(audio_16k);
                clean_audio_simple(audio_16k, MFCC_INPUT_SAMPLES);

            #if DEBUG_PLAYBACK_WINDOW
                upsample_16k_to_32k_for_playback();
                play_window_buffer();
            #endif

                compute_mfcc(audio_16k, MFCC_INPUT_SAMPLES, mfcc_buffer);

                float confidence;
                int result = tinyml_run_inference(mfcc_buffer, &confidence);

                if (result == 0 && confidence > 0.60f) {
                    printf(">>> WAKE WORD DETECTED: class=%d conf=%.4f <<<\r\n", result, confidence);
                    current_state = STATE_TASK_ACTIVE;
                } else {
                    printf("Unknown: class=%d, conf=%.4f\r\n", result, confidence);
                }
            }
        }
        else if (current_state == STATE_TASK_ACTIVE) {
            perform_wake_word_task();
            current_state = STATE_LISTENING;   // resume listening once task finishes
        }

        qcc74x_mtimer_delay_ms(5);
    }
}