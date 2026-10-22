#!/usr/bin/env python3
# Copyright (C) 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Host regressions for the decode layout; numerical kernels use GVSoC checks."""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from generate import ElfWriter, ceildiv, layout, load_arch, pack_matrix
from model import CASES, KT, NT, Model
from run import report, STAGES

APP = Path(__file__).resolve().parents[1]


class DecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = load_arch(APP.parents[1] / "apps_dram3d/config/arch.py")
        cls.model = Model()

    def test_all_presets_fit_existing_hardware(self):
        for name, (batch, seq, past) in CASES.items():
            for optimized in (False, True) if seq == 1 else (False,):
                with self.subTest(case=name, optimized=optimized):
                    plan = layout(self.model, batch, seq, past, self.arch, optimized)
                    self.assertLessEqual(plan["tcdm_bytes"], self.arch.cluster_tcdm_size)
                    self.assertLessEqual(plan["channel_bytes"], 2**31)
                    self.assertLessEqual(plan["channel_bytes"], self.arch.dram3d_node_space)
                    if optimized:
                        # Both panels, including the shorter final K panel,
                        # must fit before the output and vector buffers.
                        self.assertGreaterEqual(plan["l1"]["c"] - plan["l1"]["w"],
                                                2 * plan["weight_panel_k_tiles"] * KT * NT * 2)
                        self.assertGreaterEqual(plan["channel_bytes"] - plan["offsets"]["activation"],
                                                ceildiv(batch, plan["channels"]) *
                                                self.model.topk * self.model.intermediate * 2)

    def test_original_preload_layout_is_preserved(self):
        plan = layout(self.model, 1, 1, 1024, self.arch)
        self.assertEqual(plan["expert_stride"], 50889600)
        self.assertEqual(plan["offsets"]["x"], 410963456)
        self.assertEqual(plan["channel_bytes"], 411649536)
        self.assertEqual(plan["l1"], dict(a=4096, x=135168, w=139264, c=155648,
                                         bias=157696, tmp=157824, routes=190592))

    def test_rotated_shards_reconstruct_weights_with_padding(self):
        nc = self.arch.num_cluster_x * self.arch.num_cluster_y
        # Non-multiples exercise both K and N tails and a partial final stripe.
        k, n = KT + 3, nc * NT + 7
        matrix = (np.arange(k * n).reshape(k, n) % 257 / 32).astype("<f2")
        for expert in (0, 1, 15, 16, 127):
            shards = []
            for cid in range(nc):
                tiles = range((cid - expert) % nc, ceildiv(n, NT), nc)
                shards.append(pack_matrix(matrix, list(tiles)))
            restored = np.full((ceildiv(k, KT) * KT, ceildiv(n, NT) * NT),
                               np.nan, dtype="<f2")
            # Independently address each logical column's home and local tile.
            for j in range(ceildiv(n, NT)):
                owner, local = (j + expert) % nc, j // nc
                restored[:, j * NT:(j + 1) * NT] = shards[owner][local].reshape(-1, NT)
            np.testing.assert_array_equal(restored[:k, :n], matrix)
            self.assertTrue(np.all(restored[k:] == 0))
            self.assertTrue(np.all(restored[:, n:] == 0))

    def test_sharded_preload_can_exceed_old_segment_limit(self):
        build = APP / "build"
        build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build) as tmp:
            path = Path(tmp) / "preload.elf"
            count = 2051
            writer = ElfWriter(path, max_segments=count)
            for i in range(count):
                writer.add(0x10000000000 + i * 128, np.array([i], dtype="<u4"))
            writer.close()
            data = path.read_bytes()
            self.assertEqual(struct.unpack_from("<H", data, 56)[0], count)
            for i in (0, 2047, count - 1):
                kind, flags, off, va, pa, size, memsize, align = struct.unpack_from(
                    "<IIQQQQQQ", data, 64 + i * 56)
                self.assertEqual((kind, flags, va, pa, size, memsize, align),
                                 (1, 6, 0x10000000000 + i * 128,
                                  0x10000000000 + i * 128, 4, 4, 1))
                self.assertEqual(struct.unpack_from("<I", data, off)[0], i)

    def test_results_reject_wrong_binary_schedule(self):
        manifest = dict(batch=1, sequence=1, past_kv=1024, layer=1,
                        verified_elements=36296, decode_optimized=True)
        text = "".join(f"LLM_STAGE id={i} cycles=1\n" for i in range(len(STAGES)))
        text += ("LLM_DRAM3D_PASS batch=1 seq=1 past=1024 layer=1 "
                 "cycles_hi=0 cycles_lo=11 errors=0 checked=36296\n")
        with self.assertRaisesRegex(ValueError, "schedule"):
            report(text, manifest)
        self.assertEqual(report("LLM_SCHEDULE decode_optimized=1\n" + text, manifest)["cycles"], 11)
        manifest["decode_optimized"] = False
        self.assertEqual(report(text, manifest)["cycles"], 11)


if __name__ == "__main__":
    unittest.main()
