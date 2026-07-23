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

#include <stdint.h>

#include "flex_annotation.h"
#include "flex_runtime.h"

#define ANNOTATION_TEST_LABEL_LENGTH 333u
#define ANNOTATION_TEST_ID_BASE      0x00001000u
#define ANNOTATION_TEST_CANCEL_ID    0xfffffffeu
#define ANNOTATION_TEST_OTHER_ID     0x80000000u

#define EXPECT_RESULT(expression, expected)            \
    do                                                  \
    {                                                   \
        if ((expression) != (uint32_t)(expected))       \
        {                                               \
            failures++;                                 \
        }                                               \
    } while (0)

/*
 * HBM is shared by all clusters. Every core owns one slot, so collecting test
 * failures needs no atomic operation.
 */
volatile uint32_t annotation_test_failures
    [ARCH_NUM_CLUSTER * ARCH_NUM_CORE_PER_CLUSTER]
    __attribute__((section(".hbm")));

static void annotation_test_delay(void)
{
    volatile uint32_t value = 1;

    for (uint32_t i = 0; i < 64u; i++)
    {
        value = value * 1664525u + 1013904223u;
    }

    __asm__ volatile("" : : "r"(value) : "memory");
}

static void annotation_test_make_long_label(char *label, uint32_t cluster_id,
                                            uint32_t core_id)
{
    static const char hex[] = "0123456789abcdef";

    label[0] = '[';
    label[1] = 'c';
    label[2] = hex[(cluster_id >> 4) & 0xfu];
    label[3] = hex[cluster_id & 0xfu];
    label[4] = ':';
    label[5] = 'h';
    label[6] = hex[core_id & 0xfu];
    label[7] = ']';

    for (uint32_t i = 8; i < ANNOTATION_TEST_LABEL_LENGTH; i++)
    {
        label[i] = (char)('a' + ((i + cluster_id + core_id) % 26u));
    }

    label[ANNOTATION_TEST_LABEL_LENGTH] = '\0';
}

int main(void)
{
    uint32_t failures = 0;
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    uint32_t id = ANNOTATION_TEST_ID_BASE + core_id;
    char long_label[ANNOTATION_TEST_LABEL_LENGTH + 1u];
    const char malformed_utf8[] = {
        'm', 'a', 'l', 'f', 'o', 'r', 'm', 'e', 'd', '-',
        (char)0xc3, (char)0x28, '\0'
    };

    flex_barrier_xy_init();
    flex_global_barrier_xy();

    /* Begin every cluster from a known, empty annotation registry. */
    if (core_id == 0)
    {
        EXPECT_RESULT(annotation_register_clear_all(), 1);
    }
    flex_global_barrier_xy();

    /*
     * All cores and clusters deliberately execute this section concurrently.
     * IDs collide across clusters but not within a cluster, validating both
     * per-cluster ownership and per-initiator string staging.
     */
    EXPECT_RESULT(annotation_register_id(id), 1);
    EXPECT_RESULT(annotation_register_id(id), 0);
    EXPECT_RESULT(annotation_register_words(id, (const char *)0), 0);
    EXPECT_RESULT(annotation_register_words(id, ""), 0);

    /* Exercise and expose the default "[Annotated <id>]" label first. */
    EXPECT_RESULT(annotation_start(id), 1);
    annotation_test_delay();
    EXPECT_RESULT(annotation_stop(id), 1);

    annotation_test_make_long_label(long_label, cluster_id, core_id);
    flex_global_barrier_xy();
    EXPECT_RESULT(annotation_register_words(id, long_label), 1);

    EXPECT_RESULT(annotation_start(id), 1);
    EXPECT_RESULT(annotation_start(id), 0);
    EXPECT_RESULT(annotation_register_words(id, "mutation-while-active"), 0);
    EXPECT_RESULT(annotation_register_clear(id), 0);
    annotation_test_delay();
    EXPECT_RESULT(annotation_stop(id), 1);
    EXPECT_RESULT(annotation_stop(id), 0);

    /*
     * Invalid streamed labels must fail atomically. The subsequent interval
     * makes the preserved label observable in the model trace.
     */
    EXPECT_RESULT(annotation_register_words(id, "preserved-after-invalid"), 1);
    EXPECT_RESULT(annotation_register_words(id, "invalid\nlabel"), 0);
    EXPECT_RESULT(annotation_register_words(id, malformed_utf8), 0);
    EXPECT_RESULT(annotation_start(id), 1);
    annotation_test_delay();
    EXPECT_RESULT(annotation_stop(id), 1);
    EXPECT_RESULT(annotation_register_clear(id), 1);
    EXPECT_RESULT(annotation_register_clear(id), 0);
    EXPECT_RESULT(annotation_register_words(id, "missing-id"), 0);
    EXPECT_RESULT(annotation_start(id), 0);
    EXPECT_RESULT(annotation_stop(id), 0);

    flex_global_barrier_xy();

    /*
     * Boundary IDs and clear-all cancellation are run by one core per cluster.
     * Reusing the same IDs in every cluster further checks registry isolation.
     */
    if (core_id == 0)
    {
        EXPECT_RESULT(annotation_register_id(0u), 1);
        EXPECT_RESULT(annotation_register_words(0u, "zero-id"), 1);
        EXPECT_RESULT(annotation_start(0u), 1);
        annotation_test_delay();
        EXPECT_RESULT(annotation_stop(0u), 1);
        EXPECT_RESULT(annotation_register_clear(0u), 1);

        EXPECT_RESULT(annotation_register_id(UINT32_MAX), 1);
        EXPECT_RESULT(annotation_start(UINT32_MAX), 1);
        annotation_test_delay();
        EXPECT_RESULT(annotation_stop(UINT32_MAX), 1);
        EXPECT_RESULT(annotation_register_clear(UINT32_MAX), 1);

        EXPECT_RESULT(annotation_register_id(ANNOTATION_TEST_CANCEL_ID), 1);
        EXPECT_RESULT(annotation_register_words(ANNOTATION_TEST_CANCEL_ID,
                                                "clear-all-active"),
                      1);
        EXPECT_RESULT(annotation_register_id(ANNOTATION_TEST_OTHER_ID), 1);
        EXPECT_RESULT(annotation_start(ANNOTATION_TEST_CANCEL_ID), 1);
        EXPECT_RESULT(annotation_register_clear_all(), 1);
        EXPECT_RESULT(annotation_stop(ANNOTATION_TEST_CANCEL_ID), 0);
        EXPECT_RESULT(annotation_register_clear(ANNOTATION_TEST_CANCEL_ID), 0);
        EXPECT_RESULT(annotation_register_clear(ANNOTATION_TEST_OTHER_ID), 0);
        EXPECT_RESULT(annotation_register_clear_all(), 1);
    }

    flex_global_barrier_xy();

    annotation_test_failures
        [cluster_id * ARCH_NUM_CORE_PER_CLUSTER + core_id] = failures;
    __asm__ volatile("fence rw, rw" ::: "memory");
    flex_global_barrier_xy();

    if (cluster_id == 0 && core_id == 0)
    {
        uint32_t total_failures = 0;

        for (uint32_t i = 0;
             i < ARCH_NUM_CLUSTER * ARCH_NUM_CORE_PER_CLUSTER; i++)
        {
            total_failures += annotation_test_failures[i];
        }

        if (total_failures == 0)
        {
            flex_print("ANNOTATION_TEST_PASS\n");
        }
        else
        {
            flex_print("ANNOTATION_TEST_FAIL failures=");
            flex_print_int(total_failures);
            flex_print("\n");
        }

        flex_eoc(total_failures == 0 ? 0u : 1u);
    }

    return 0;
}
