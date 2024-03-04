#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir import to_edge
import torch
from torch.export import export

from torch.nn.attention import SDPBackend
from model import GPT, GPTConfig
from torch._export import capture_pre_autograd_graph
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, ExecutorchProgramManager

OUTPUT_FILE = "nanogpt.pte" # Output ExecuTorch program file.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model = GPT(GPTConfig())
example_inputs = (torch.randint(0, 100, (1, 1024), dtype=torch.long), )

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Export  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Using a custom SDPA kernel for LLMs
with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    m = capture_pre_autograd_graph( model, example_inputs)
    edge_config = EdgeCompileConfig(_check_ir_validity=False)

    print("Exporting program...")
    core_aten_ep = export(m, example_inputs)
    print("Lowering to edge...")
    edge_manager: EdgeProgramManager = to_edge(
        core_aten_ep,
        compile_config=edge_config,
    )
    print("Creating ExecuTorch program...")
    et_program: ExecutorchProgramManager = edge_manager.to_executorch()

# Write the serialized ExecuTorch program to a file.
with open(OUTPUT_FILE, "wb") as file:
    file.write(et_program.buffer)
    print(f"ExecuTorch program saved to {OUTPUT_FILE}.")
