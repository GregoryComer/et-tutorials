#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.tracer import Value
import torch
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram

from simple_model import SimpleModel

STATE_FILE = "SimpleModel.pt" # Contains trained parameters.
OUTPUT_FILE = "SimpleModel.pte" # Output ExecuTorch program file.

# Instantiate our model.
model = SimpleModel()

# This helps the exporter understand the model shape.
# Since our model only takes one input, this is a one-tuple.
example_inputs = (torch.ones(4),)

# Export the pytorch model and process for ExecuTorch.
print("Exporting program...")
exported_program = export(model, example_inputs)
print("Lowering to edge...")
edge_program = to_edge(exported_program)
print("Creating ExecuTorch program...")
et_program = edge_program.to_executorch()

# Write the serialized ExecuTorch program to a file.
with open(OUTPUT_FILE, "wb") as file:
    file.write(et_program.buffer)
    print(f"ExecuTorch program saved to {OUTPUT_FILE}.")
