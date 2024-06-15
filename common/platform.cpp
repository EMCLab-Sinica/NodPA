#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include "c_callbacks.h"
#include "config.h"
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "platform.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "intermittent-cnn.h" // for sample_idx
#include "double_buffering.h"

// put offset checks here as extra headers are used
static_assert(COUNTERS_OFFSET >= PARAMETERS_OFFSET + PARAMETERS_DATA_LEN, "Incorrect NVM layout");

Model model_vm;
const uint8_t NUM_PARAMETER_INFO_SLOTS = 1 + NUM_INPUTS; // ParameterInfo for one output and several inputs
static ParameterInfo intermediate_parameters_info_vm[NUM_PARAMETER_INFO_SLOTS];

static uint32_t intermediate_values_offset(uint8_t slot_id) {
    return INTERMEDIATE_VALUES_OFFSET + slot_id * INTERMEDIATE_VALUES_SIZE;
}

static uint32_t intermediate_parameters_info_addr(uint16_t i) {
    return INTERMEDIATE_PARAMETERS_INFO_OFFSET + i * sizeof(ParameterInfo);
}

template<>
uint32_t nvm_addr<Model>(uint8_t i, uint16_t) {
    return MODEL_OFFSET + i * sizeof(Model);
}

template<>
Model* vm_addr<Model>(uint16_t data_idx) {
    return &model_vm;
}

template<>
const char* datatype_name<Model>(void) {
    return "model";
}

static void notify_progress(void) {
#if 0
    // indicate there is some progress in this power cycle
    static bool notified = false;
    if (!notified) {
        notify_indicator(1);
        notified = true;
    }
#endif
}

void my_memcpy_to_param(ParameterInfo *param, uint32_t offset_in_word, const void *src, size_t n, uint16_t timer_delay, bool is_linear) {
    MY_ASSERT(param->slot < NUM_SLOTS);
    uint32_t total_offset = param->params_offset + offset_in_word * sizeof(int16_t);

#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_FINE_GRAINED
    MY_ASSERT(total_offset + n <= param->params_len);
#else
    if (total_offset + n > param->params_len) {
        return;
    }
#endif

#if ENABLE_COUNTERS
    uint32_t n_jobs;
#if JAPARI
    uint16_t n_footprints = n / (BATCH_SIZE + 1);
    n_jobs = n - n_footprints;
    add_counter(offsetof(Counters, nvm_write_footprints), n_footprints);
    my_printf_debug("Recorded %u bytes of footprints written to NVM" NEWLINE, n_footprints);
#else
    n_jobs = n;
#endif // JAPARI
    if (is_linear) {
        add_counter(offsetof(Counters, nvm_write_linear_jobs), n_jobs);
        my_printf_debug("Recorded %u bytes of linear jobs written to NVM" NEWLINE, n_jobs);
    } else {
        add_counter(offsetof(Counters, nvm_write_non_linear_jobs), n_jobs);
        my_printf_debug("Recorded %u bytes of non-linear jobs written to NVM" NEWLINE, n_jobs);
    }
#endif

    write_to_nvm(src, intermediate_values_offset(param->slot) + total_offset, n, timer_delay);

    notify_progress();
}

void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint32_t offset_in_word, size_t n) {
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_job_outputs), n);
        my_printf_debug("Recorded %lu bytes of job outputs fetched from NVM, accumulated=%" PRIu32 NEWLINE, n, get_counter(offsetof(Counters, nvm_read_job_outputs)));
    }
#endif

    read_from_nvm(dest, intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t), n);
}

void read_from_samples(void *dest, uint32_t offset_in_word, size_t n) {
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_job_outputs), n);
        my_printf_debug("Recorded %lu bytes of samples fetched from NVM, accumulated=%" PRIu32 NEWLINE, n, get_counter(offsetof(Counters, nvm_read_job_outputs)));
    }
#endif

    read_from_nvm(dest, SAMPLES_OFFSET + (inference_results_vm.sample_idx % LABELS_DATA_LEN) * 2*TOTAL_SAMPLE_SIZE + offset_in_word * sizeof(int16_t), n);
}

static uint8_t get_available_parameter_info_slot() {
    for (uint8_t parameter_info_slot_idx = 0; parameter_info_slot_idx < NUM_PARAMETER_INFO_SLOTS; parameter_info_slot_idx++) {
        if (intermediate_parameters_info_vm[parameter_info_slot_idx].parameter_info_idx == 0) {
            return parameter_info_slot_idx;
        }
    }

    MY_ASSERT(false);

    return UINT8_MAX;
}

ParameterInfo* get_intermediate_parameter_info(uint16_t i) {
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_read_model), sizeof(ParameterInfo));
        my_printf_debug("Recorded %lu bytes of ParameterInfo fetched from NVM" NEWLINE, sizeof(ParameterInfo));
    }
#endif
    ParameterInfo* dst = intermediate_parameters_info_vm + get_available_parameter_info_slot();
    read_from_nvm(dst, intermediate_parameters_info_addr(i), sizeof(ParameterInfo));
    my_printf_debug("Load intermediate parameter info %d from NVM" NEWLINE, i);
    MY_ASSERT(dst->parameter_info_idx == i + N_INPUT,
              "Expect parameter index %d but got %d" NEWLINE, i + N_INPUT, dst->parameter_info_idx);
    return dst;
}

void commit_intermediate_parameter_info(const ParameterInfo* param) {
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    if (counters_enabled) {
        add_counter(offsetof(Counters, nvm_write_model), sizeof(ParameterInfo));
        my_printf_debug("Recorded %lu bytes of ParameterInfo written NVM" NEWLINE, sizeof(ParameterInfo));
    }
#endif
    uint16_t node_idx = param->parameter_info_idx - N_INPUT;
    write_to_nvm(param, intermediate_parameters_info_addr(node_idx), sizeof(ParameterInfo));
    my_printf_debug("Committing intermediate parameter info %d to NVM" NEWLINE, node_idx);
}

void flush_intermediate_parameter_info() {
    for (uint8_t parameter_info_slot_idx = 0; parameter_info_slot_idx < NUM_PARAMETER_INFO_SLOTS; parameter_info_slot_idx++) {
        intermediate_parameters_info_vm[parameter_info_slot_idx].parameter_info_idx = 0;
    }
}

Model* load_model_from_nvm(void) {
    start_cpu_counter(offsetof(Counters, table_loading));
    Model* ret = get_versioned_data<Model>(0);
    stop_cpu_counter();
    return ret;
}

Model* get_model(void) {
    return &model_vm;
}

void commit_model(void) {
#if ENABLE_DEMO_COUNTERS
    if (!model_vm.running) {
        reset_counters();
    }
#endif
    start_cpu_counter(offsetof(Counters, table_preservation));
    commit_versioned_data<Model>(0);
    // send finish signals only after the whole network has really finished
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    add_counter(offsetof(Counters, power_counters), 1);
#endif
    if (!model_vm.running) {
        notify_model_finished();
    }
    stop_cpu_counter();
}

void first_run(void) {
    my_printf_debug("First run, resetting everything..." NEWLINE);
    disable_counters();
    my_erase();
    copy_data_to_nvm();
    reset_counters();

    write_to_nvm_segmented(intermediate_parameters_info_data, intermediate_parameters_info_addr(0),
                           INTERMEDIATE_PARAMETERS_INFO_DATA_LEN, sizeof(ParameterInfo));
    write_to_nvm(model_data, nvm_addr<Model>(0, 0), MODEL_DATA_LEN);
    write_to_nvm(model_data, nvm_addr<Model>(1, 0), MODEL_DATA_LEN);

    load_model_from_nvm(); // refresh model_vm
    commit_model();

    my_printf_debug("Init for " CONFIG "/" METHOD " with batch size=%d" NEWLINE, BATCH_SIZE);
    enable_counters();
}

void read_from_nvm_segmented(uint8_t* vm_buffer, uint32_t nvm_offset, uint32_t total_len, uint16_t segment_size) {
    for (uint32_t idx = 0; idx < total_len; idx += segment_size) {
        read_from_nvm(vm_buffer + idx, nvm_offset + idx, MIN_VAL(total_len - idx, segment_size));
    }
}

void write_to_nvm_segmented(const uint8_t* vm_buffer, uint32_t nvm_offset, uint32_t total_len, uint16_t segment_size) {
    for (uint32_t idx = 0; idx < total_len; idx += segment_size) {
        write_to_nvm(vm_buffer + idx, nvm_offset + idx, MIN_VAL(total_len - idx, segment_size));
    }
}

void record_overflow_handling_overhead(uint32_t cycles) {
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    add_counter(offsetof(Counters, overflow_handling), cycles);
#endif
}

#if HAWAII
NOINIT static Footprint footprints_vm[MODEL_NODES_LEN];
NOINIT static FootprintForDynamicDNN footprints_for_dynamic_dnn_vm[MODEL_NODES_LEN];
static uint8_t footprint_copy_id = 0;
static uint8_t footprint_for_dynamic_dnn_copy_id = 0;

template<>
uint32_t nvm_addr<Footprint>(uint8_t copy_id, uint16_t layer_idx) {
    return FOOTPRINTS_OFFSET + (copy_id * MODEL_NODES_LEN + layer_idx) * sizeof(Footprint);
}

template<>
uint32_t nvm_addr<FootprintForDynamicDNN>(uint8_t copy_id, uint16_t layer_idx) {
    return FOOTPRINTS_FOR_DYNAMIC_DNN_OFFSET + (copy_id * MODEL_NODES_LEN + layer_idx) * sizeof(FootprintForDynamicDNN);
}

template<>
Footprint* vm_addr<Footprint>(uint16_t layer_idx) {
    return &footprints_vm[layer_idx];
}

template<>
FootprintForDynamicDNN* vm_addr<FootprintForDynamicDNN>(uint16_t layer_idx) {
    return &footprints_for_dynamic_dnn_vm[layer_idx];
}

template<>
const char* datatype_name<Footprint>(void) {
    return "footprint";
}

template<>
const char* datatype_name<FootprintForDynamicDNN>(void) {
    return "dynamic information";
}

template<>
uint8_t* copy_id_cache_addr<Footprint>(void) {
    return &footprint_copy_id;
}

template<>
uint8_t* copy_id_cache_addr<FootprintForDynamicDNN>(void) {
    return &footprint_for_dynamic_dnn_copy_id;
}

template<typename T>
void write_hawaii_layer_footprint_impl(uint16_t layer_idx, int16_t value) {
#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_COARSE_GRAINED
    T* footprint_vm = vm_addr<T>(layer_idx);

#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_TWO_INDICATOR
    footprint_vm->value += value;
#else
    uint32_t old_value = footprint_vm->value;
    footprint_vm->value += value;

    // Comparing old and new values. If only the least significant byte is changed, update that byte only.
    // Otherwise, update the whole value via double buffering.
    constexpr decltype(T::value) mask = std::numeric_limits<decltype(T::value)>::max() - 0xff;
    if ((footprint_vm->value & mask) == (old_value & mask)) {
        uint8_t* copy_id_cache = copy_id_cache_addr<T>();
        MY_ASSERT(copy_id_cache && *copy_id_cache);

        uint32_t footprint_nvm_addr = nvm_addr<T>(*copy_id_cache - 1, layer_idx);
        // Calculate the address for the least significant byte according to system endianness
        // Note that __BYTE_ORDER__ here is a GCC extension
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        uint32_t last_byte_nvm_addr = footprint_nvm_addr + offsetof(T, value) + sizeof(T::value) - 1;
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        uint32_t last_byte_nvm_addr = footprint_nvm_addr + offsetof(T, value);
#else
#error "Unsupported endian"
#endif
        uint8_t last_byte = static_cast<uint8_t>(footprint_vm->value & 0xff);
        write_to_nvm(&last_byte, last_byte_nvm_addr, /*n=*/1);

        my_printf_debug("Update the least significant byte (%x) for HAWAII layer %s %d for layer %d" NEWLINE,
                        last_byte, datatype_name<T>(), footprint_vm->value, layer_idx);
    } else
#endif
    {
#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_TWO_INDICATOR_NAIVE
        commit_versioned_data<Footprint>(layer_idx);
        commit_versioned_data<FootprintForDynamicDNN>(layer_idx);
        my_printf_debug("Commit HAWAII layer footprint and dynamic information for layer %d" NEWLINE, layer_idx);
#else
        commit_versioned_data<T>(layer_idx);
        my_printf_debug("Commit HAWAII layer %s %d for layer %d" NEWLINE, datatype_name<T>(), footprint_vm->value, layer_idx);
#endif
    }
    MY_ASSERT(footprint_vm->value % BATCH_SIZE == 0);
#endif
}

void write_hawaii_layer_footprint(uint16_t layer_idx, int16_t n_jobs) {
    write_hawaii_layer_footprint_impl<Footprint>(layer_idx, n_jobs);
}

void write_hawaii_dynamic_dnn_information(uint16_t layer_idx, uint32_t value) {
    write_hawaii_layer_footprint_impl<FootprintForDynamicDNN>(layer_idx, value);
}

template<typename T>
uint32_t read_hawaii_layer_footprint(uint16_t layer_idx) {
    const T* footprint = get_versioned_data<T>(layer_idx);
    my_printf_debug("HAWAII layer %s=%d for layer %d" NEWLINE, datatype_name<T>(), footprint->value, layer_idx);
    MY_ASSERT(footprint->value % BATCH_SIZE == 0);
    return footprint->value;
}

template<typename T>
void reset_hawaii_layer_footprint(uint16_t layer_idx) {
    T footprint;
    memset(&footprint, 0, sizeof(T));
    write_to_nvm(&footprint, nvm_addr<T>(0, layer_idx), sizeof(T));
    write_to_nvm(&footprint, nvm_addr<T>(1, layer_idx), sizeof(T));
    my_printf_debug("Reset HAWAII layer %s for layer %d" NEWLINE, datatype_name<T>(), layer_idx);
}

// Explicit instantiations
template uint32_t read_hawaii_layer_footprint<Footprint>(uint16_t layer_idx);
template uint32_t read_hawaii_layer_footprint<FootprintForDynamicDNN>(uint16_t layer_idx);
template void reset_hawaii_layer_footprint<Footprint>(uint16_t layer_idx);
template void reset_hawaii_layer_footprint<FootprintForDynamicDNN>(uint16_t layer_idx);

#endif
