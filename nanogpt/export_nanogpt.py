#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir import to_edge
import torch
from torch.export import export

from model import GPT, GPTConfig

OUTPUT_FILE = "nanogpt.pte" # Output ExecuTorch program file.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Default Config
config = GPTConfig()

# GPTConfig from train.py - scratch Input when printing (64, 256)
config_train = GPTConfig(block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.2, bias=False)

model = GPT(config)

"""
for layer in model.children():
    print(layer)
    print(type(layer))
    for name, layer in layer.items():
        print(name, layer)
"""

example_inputs = (torch.ones(50304, 768, dtype=torch.int8), )

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Export  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
