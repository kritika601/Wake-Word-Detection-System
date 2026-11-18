cmake_minimum_required(VERSION 3.15)

include(proj.conf)

find_package(qcc74x_sdk REQUIRED HINTS $ENV{QCC74x_SDK_BASE})

sdk_add_compile_definitions(-DCONFIG_CLI_CMD_ENABLE)

if(CONFIG_AUDIO_MIC_PDM_MODE)
    sdk_add_compile_definitions(-DCONFIG_AUDIO_MIC_PDM_MODE)
endif()


sdk_add_compile_definitions(
    TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
    TF_LITE_USE_GLOBAL_MIN
    TF_LITE_USE_GLOBAL_MAX
    TF_LITE_MCU_DEBUG_LOG
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

sdk_add_include_directories(
    .
    third_party
    third_party/tensorflow
    third_party/third_party/flatbuffers/include
    third_party/third_party/gemmlowp
    third_party/third_party/kissfft
    third_party/third_party/ruy
)

# --------------------------------------------------------------------
# TensorFlow Lite Micro sources
# --------------------------------------------------------------------

file(GLOB_RECURSE TFLM_CPP
    third_party/tensorflow/*.cpp
)

file(GLOB_RECURSE TFLM_C
    third_party/tensorflow/*.c
)

# message(STATUS "========== TFLM SOURCES ==========")
# foreach(f ${TFLM_CPP})
#     message(STATUS "${f}")
# endforeach()

# Remove tests
list(FILTER TFLM_CPP EXCLUDE REGEX ".*test.*\\.cpp$")
list(FILTER TFLM_CPP EXCLUDE REGEX ".*mock.*\\.cpp$")
list(FILTER TFLM_CPP EXCLUDE REGEX ".*fake.*\\.cpp$")

# Remove benchmark/example files
list(FILTER TFLM_CPP EXCLUDE REGEX ".*/benchmark/.*")
# list(FILTER TFLM_CPP EXCLUDE REGEX ".*/examples/.*")

# Remove host-only files
list(FILTER TFLM_CPP EXCLUDE REGEX ".*/micro_time\\.cpp$")
list(FILTER TFLM_CPP EXCLUDE REGEX ".*/system_setup\\.cpp$")
list(FILTER TFLM_CPP EXCLUDE REGEX ".*/micro/debug_log\\.cpp$")

sdk_set_main_file(main.c)

target_sources(app PRIVATE
    tinyml.cpp
    cpp_support.cpp
    tinyml_debug.cpp
    audio_features.cpp
    model_data.c
    third_party/third_party/kissfft/kiss_fft.c
    third_party/third_party/kissfft/tools/kiss_fftr.c
    
    

    ${TFLM_CPP}
    ${TFLM_C}
)

set_source_files_properties(
    third_party/third_party/kissfft/kiss_fft.c
    third_party/third_party/kissfft/tools/kiss_fftr.c
    PROPERTIES LANGUAGE CXX
)

project(project_name)