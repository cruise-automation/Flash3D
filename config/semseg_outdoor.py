#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/14/24


from torch import nn
from functools import partial
import transformer_engine.pytorch as te

import flash3dxfmr as f3d
from flash3dxfmr.psh import bucket_scope as bksp


buck_size = 512
scope_size = 8
hash_op = 2

nl = te.LayerNorm
act = nn.GELU
pln = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

outdoor_specs = [
    # Hourglass Stage One
    f3d.F3DLevelSpecs(
        encoder_specs=[
            f3d.XFMRSpecs(
                channels=32,
                hid_channels=128,
                num_heads=2,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 1),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=32,
                hid_channels=128,
                num_heads=2,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("stride", buck_size, scope_size, 2),
                norm_layer=nl
            )
        ],
        decoder_specs=[
            f3d.XFMRSpecs(
                channels=64,
                hid_channels=256,
                num_heads=4,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 3),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=64,
                hid_channels=256,
                num_heads=4,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 5),
                norm_layer=nl
            )
        ],
        reduction_op="mean",
        pool_align=bksp.swinable_alignment(buck_size, scope_size),
        pool_norm=pln,
        pool_act=act,
        unpool_norm=pln,
        unpool_act=act
    ),

    # Hourglass Stage Two
    f3d.F3DLevelSpecs(
        encoder_specs=[
            f3d.XFMRSpecs(
                channels=64,
                hid_channels=256,
                num_heads=4,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 7),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=64,
                hid_channels=256,
                num_heads=4,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 2),
                norm_layer=nl
            )
        ],
        decoder_specs=[
            f3d.XFMRSpecs(
                channels=64,
                hid_channels=256,
                num_heads=4,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 4),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=64,
                hid_channels=256,
                num_heads=4,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 6),
                norm_layer=nl
            )
        ],
        reduction_op="mean",
        pool_align=bksp.swinable_alignment(buck_size, scope_size),
        pool_norm=pln,
        pool_act=act,
        unpool_norm=pln,
        unpool_act=act
    ),

    # Hourglass Stage Three
    f3d.F3DLevelSpecs(
        encoder_specs=[
            f3d.XFMRSpecs(
                channels=128,
                hid_channels=512,
                num_heads=8,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 1),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=128,
                hid_channels=512,
                num_heads=8,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 5),
                norm_layer=nl
            )
        ],
        decoder_specs=[
            f3d.XFMRSpecs(
                channels=128,
                hid_channels=512,
                num_heads=8,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 2),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=128,
                hid_channels=512,
                num_heads=8,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 6),
                norm_layer=nl
            )
        ],
        reduction_op="mean",
        pool_align=bksp.swinable_alignment(buck_size, scope_size),
        pool_norm=pln,
        pool_act=act,
        unpool_norm=pln,
        unpool_act=act
    ),

    # Hourglass Stage Four
    f3d.F3DLevelSpecs(
        encoder_specs=[
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,
                num_heads=16,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 3),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,
                num_heads=16,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 7),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,
                num_heads=16,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 0),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,
                num_heads=16,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 6),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,
                num_heads=16,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 4),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,
                num_heads=16,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 2),
                norm_layer=nl
            )
        ],
        decoder_specs=[
            f3d.XFMRSpecs(
                channels=128,
                hid_channels=512,   # 4 * 128
                num_heads=8,        # 128 / 16
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 0),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=128,
                hid_channels=512,
                num_heads=8,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 3),
                norm_layer=nl
            )
        ],
        reduction_op="mean",
        pool_align=bksp.swinable_alignment(buck_size, scope_size),
        pool_norm=pln,
        pool_act=act,
        unpool_norm=pln,
        unpool_act=act
    ),

    # Hourglass Stage Five
    f3d.F3DLevelSpecs(
        encoder_specs=[
            f3d.XFMRSpecs(
                channels=512,
                hid_channels=2048,  # 4 * 512
                num_heads=32,       # 512 / 16
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 5),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=512,
                hid_channels=2048,
                num_heads=32,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 7),
                norm_layer=nl
            )
        ],
        decoder_specs=[
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,  # 4 * 256
                num_heads=16,       # 256 / 16
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 1),
                norm_layer=nl
            ),
            f3d.XFMRSpecs(
                channels=256,
                hid_channels=1024,
                num_heads=16,
                qkv_bias=True,
                swin_plan=bksp.SwinPlan("swin", buck_size, scope_size, 5),
                norm_layer=nl
            )
        ],
        reduction_op="mean",
        pool_align=bksp.swinable_alignment(buck_size, scope_size),
        pool_norm=pln,
        pool_act=act,
        unpool_norm=pln,
        unpool_act=act
    )


]