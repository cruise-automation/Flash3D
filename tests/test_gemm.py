#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 1/15/25
#

import torch
import unittest
from flash3dxfmr.lib import pshattn
from flash3dxfmr.psh import dev_context

dev_context.set_sync_stream(True)


def measure_error_stats(pred, label):
    abs_diff = torch.abs(pred - label)
    max_err = abs_diff.max().item()
    avg_err = abs_diff.mean().item()

    return max_err, avg_err


class TestGEMM(unittest.TestCase):
    def test_256_16_k_bf16(self):
        a = torch.randn(128, 128, dtype=torch.bfloat16, device=0)
        b = torch.randn(128, 16, dtype=torch.bfloat16, device=0)
        o = torch.randn(128, 16, dtype=torch.bfloat16, device=0)

        pshattn.gemm_sm_bf16(a, b, o)
        max_err, avg_err = measure_error_stats(a @ b, o)

        self.assertTrue(
            torch.allclose(a @ b, o, atol=1e-3), 
            f"Custom GEMM(bf16) differs from cuBLAS results. max_err={max_err},avg_err={avg_err}")


if __name__ == "__main__":
    unittest.main()
