//
// Copyright (C) 2026 ETH Zurich and University of Bologna
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef _FLEX_ANNOTATION_H_
#define _FLEX_ANNOTATION_H_

#include <stdint.h>

#include "flex_cluster_arch.h"

/*
 * Annotation register protocol.
 *
 * All registers are 32-bit and local to the calling core's cluster. String
 * bytes are packed little-endian into DATA, four bytes per MMIO access. LENGTH
 * excludes the terminating NUL.
 */
#define FLEX_ANNOTATION_COMMAND_OFFSET 0x000001c0u
#define FLEX_ANNOTATION_ID_OFFSET      0x000001c4u
#define FLEX_ANNOTATION_LENGTH_OFFSET  0x000001c8u
#define FLEX_ANNOTATION_DATA_OFFSET    0x000001ccu
#define FLEX_ANNOTATION_RESULT_OFFSET  0x000001d0u
#define FLEX_ANNOTATION_LOCK_OFFSET    0x000001d4u

#define FLEX_ANNOTATION_COMMAND_REGISTER_ID    1u
#define FLEX_ANNOTATION_COMMAND_REGISTER_WORDS 2u
#define FLEX_ANNOTATION_COMMAND_CLEAR          3u
#define FLEX_ANNOTATION_COMMAND_CLEAR_ALL      4u
#define FLEX_ANNOTATION_COMMAND_START          5u
#define FLEX_ANNOTATION_COMMAND_STOP           6u

static inline void annotation_mmio_fence(void)
{
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static inline void annotation_mmio_write(uint32_t offset, uint32_t value)
{
    volatile uint32_t *reg =
        (volatile uint32_t *)(uintptr_t)(ARCH_CLUSTER_REG_BASE + offset);

    annotation_mmio_fence();
    *reg = value;
    annotation_mmio_fence();
}

static inline uint32_t annotation_mmio_read(uint32_t offset)
{
    volatile uint32_t *reg =
        (volatile uint32_t *)(uintptr_t)(ARCH_CLUSTER_REG_BASE + offset);

    annotation_mmio_fence();
    uint32_t value = *reg;
    annotation_mmio_fence();
    return value;
}

static inline void annotation_lock_acquire(void)
{
    while (!annotation_mmio_read(FLEX_ANNOTATION_LOCK_OFFSET))
    {
    }
}

static inline void annotation_lock_release(void)
{
    /* The model treats any write to LOCK as a release. */
    annotation_mmio_write(FLEX_ANNOTATION_LOCK_OFFSET, 0u);
}

/* Caller must hold the annotation lock for the complete transaction. */
static inline uint32_t annotation_command_with_id_locked(uint32_t command,
                                                         uint32_t id)
{
    annotation_mmio_write(FLEX_ANNOTATION_ID_OFFSET, id);
    annotation_mmio_write(FLEX_ANNOTATION_COMMAND_OFFSET, command);
    return annotation_mmio_read(FLEX_ANNOTATION_RESULT_OFFSET);
}

/*
 * Scan a valid C string without imposing an artificial annotation-size limit.
 * A UINT32_MAX-byte string is representable by the MMIO protocol; anything
 * longer is not. As with every strlen-like operation, the caller must provide
 * readable, stable, NUL-terminated storage. A general invalid non-NULL pointer
 * cannot be detected by C code before dereferencing it.
 */
static inline uint32_t annotation_string_length(const char *str,
                                                uint32_t *length)
{
    uint32_t current = 0;

    if (str == (const char *)0 || str[0] == '\0')
    {
        return 0;
    }

    while (str[current] != '\0')
    {
        if (current == UINT32_MAX)
        {
            return 0;
        }
        current++;
    }

    *length = current;
    return 1;
}

static inline uint32_t annotation_register_id(uint32_t id)
{
    uint32_t result;

    annotation_lock_acquire();
    result = annotation_command_with_id_locked(
        FLEX_ANNOTATION_COMMAND_REGISTER_ID, id);
    annotation_lock_release();
    return result;
}

static inline uint32_t annotation_register_words(uint32_t id, const char *str)
{
    uint32_t length;
    uint32_t position = 0;

    if (!annotation_string_length(str, &length))
    {
        return 0;
    }

    annotation_lock_acquire();
    annotation_mmio_write(FLEX_ANNOTATION_ID_OFFSET, id);
    annotation_mmio_write(FLEX_ANNOTATION_LENGTH_OFFSET, length);
    annotation_mmio_write(FLEX_ANNOTATION_COMMAND_OFFSET,
                          FLEX_ANNOTATION_COMMAND_REGISTER_WORDS);

    /*
     * The first result read is a preflight check. In particular, it avoids
     * streaming the string when the ID is not registered or is active.
     */
    if (!annotation_mmio_read(FLEX_ANNOTATION_RESULT_OFFSET))
    {
        annotation_lock_release();
        return 0;
    }

    while (position < length)
    {
        uint32_t remaining = length - position;
        uint32_t chunk_size = remaining < 4u ? remaining : 4u;
        uint32_t word = 0;

        for (uint32_t byte = 0; byte < chunk_size; byte++)
        {
            word |= ((uint32_t)(uint8_t)str[position + byte]) << (byte * 8u);
        }

        annotation_mmio_write(FLEX_ANNOTATION_DATA_OFFSET, word);
        position += chunk_size;
    }

    uint32_t result = annotation_mmio_read(FLEX_ANNOTATION_RESULT_OFFSET);
    annotation_lock_release();
    return result;
}

static inline uint32_t annotation_register_clear(uint32_t id)
{
    uint32_t result;

    annotation_lock_acquire();
    result = annotation_command_with_id_locked(FLEX_ANNOTATION_COMMAND_CLEAR,
                                               id);
    annotation_lock_release();
    return result;
}

static inline uint32_t annotation_register_clear_all(void)
{
    uint32_t result;

    annotation_lock_acquire();
    annotation_mmio_write(FLEX_ANNOTATION_COMMAND_OFFSET,
                          FLEX_ANNOTATION_COMMAND_CLEAR_ALL);
    result = annotation_mmio_read(FLEX_ANNOTATION_RESULT_OFFSET);
    annotation_lock_release();
    return result;
}

static inline uint32_t annotation_start(uint32_t id)
{
    uint32_t result;

    annotation_lock_acquire();
    result = annotation_command_with_id_locked(FLEX_ANNOTATION_COMMAND_START,
                                               id);
    annotation_lock_release();
    return result;
}

static inline uint32_t annotation_stop(uint32_t id)
{
    uint32_t result;

    annotation_lock_acquire();
    result = annotation_command_with_id_locked(FLEX_ANNOTATION_COMMAND_STOP,
                                               id);
    annotation_lock_release();
    return result;
}

#endif // _FLEX_ANNOTATION_H_
