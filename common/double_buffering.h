#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "my_debug.h"

// Templates to be filled by users
template<typename T>
static uint32_t nvm_addr(uint8_t, uint16_t);

template<typename T>
T* vm_addr(uint16_t data_idx);

// typeinfo does not always give names I want
template<typename T>
const char* datatype_name(void);

template<typename T>
uint8_t* copy_id_cache_addr(void) {
    // No copy_id cache by default
    return nullptr;
}

template<typename T>
static uint8_t get_newer_copy_id(uint16_t data_idx) {
    uint8_t* copy_id_cache = copy_id_cache_addr<T>();
    // Use 1 for copy 0 and 2 for copy 1. 0, which comes after power resumption, is considerd invalid (not cached yet)
    if (copy_id_cache && *copy_id_cache > 0) {
        return *copy_id_cache - 1;
    }

    uint8_t version1, version2;
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    add_counter(offsetof(Counters, nvm_read_shadow_data), 2);
#endif
    read_from_nvm(&version1, nvm_addr<T>(0, data_idx) + offsetof(T, version), sizeof(uint8_t));
    read_from_nvm(&version2, nvm_addr<T>(1, data_idx) + offsetof(T, version), sizeof(uint8_t));
    my_printf_debug("Versions of shadow %s copies for data item %d: %d, %d" NEWLINE, datatype_name<T>(), data_idx, version1, version2);

    uint8_t copy_id;
    if (abs(static_cast<int>(version1 - version2)) == 1) {
        if (version1 > version2) {
            copy_id = 0;
        } else {
            copy_id = 1;
        }
    } else {
        if (version1 > version2) {
            // ex: versions = 65535, 1
            copy_id = 1;
        } else {
            copy_id = 0;
        }
    }

    if (copy_id_cache) {
        *copy_id_cache = copy_id + 1;
    }

    return copy_id;
}

template<typename T>
void bump_version(T *data) {
    data->version++;
    if (!data->version) {
        // don't use version 0 as it indicates the first run
        data->version++;
    }
}

template<typename T>
T* get_versioned_data(uint16_t data_idx) {
    T *dst = vm_addr<T>(data_idx);

    uint8_t newer_copy_id = get_newer_copy_id<T>(data_idx);
#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    add_counter(offsetof(Counters, nvm_read_shadow_data), sizeof(T));
    my_printf_debug("Recorded %lu bytes of shadow data read from NVM" NEWLINE, sizeof(T));
#endif
    read_from_nvm(dst, nvm_addr<T>(newer_copy_id, data_idx), sizeof(T));
    my_printf_debug("Using %s copy %d, version %d" NEWLINE, datatype_name<T>(), newer_copy_id, dst->version);
    return dst;
}

template<typename T>
void commit_versioned_data(uint16_t data_idx) {
    uint8_t newer_copy_id = get_newer_copy_id<T>(data_idx);
    uint8_t older_copy_id = newer_copy_id ^ 1;

    T* vm_ptr = vm_addr<T>(data_idx);
    bump_version<T>(vm_ptr);

#if ENABLE_COUNTERS && !ENABLE_DEMO_COUNTERS
    add_counter(offsetof(Counters, nvm_write_shadow_data), sizeof(T));
    my_printf_debug("Recorded %lu bytes of shadow data written to NVM" NEWLINE, sizeof(T));
#endif
    write_to_nvm(vm_ptr, nvm_addr<T>(older_copy_id, data_idx), sizeof(T));
    my_printf_debug("Committing version %d to %s copy %d" NEWLINE, vm_ptr->version, datatype_name<T>(), older_copy_id);

    uint8_t* copy_id_cache = copy_id_cache_addr<T>();
    if (copy_id_cache) {
        *copy_id_cache = older_copy_id + 1;
    }
}
