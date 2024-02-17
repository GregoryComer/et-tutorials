/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <cstdio>

using namespace ::torch::executor;

const char* kDefaultProgramPath = "SimpleModel.pte";

int main(int argc, const char** argv) {
  const char* program_path = kDefaultProgramPath;
  if (argc == 2) {
    program_path = argv[1];
  } else if (argc > 2) {
    fprintf(stderr, "Usage: %s [program_file]\n", argv[0]);
    return -1;
  }

  // Load the model .pte file created by the export process. The Module type
  // encapsulates the loaded model.
  printf("Loading ExecuTorch program from %s...\n", program_path);
  Module module(program_path);

  // Set up the model input data. This model takes a 1-dimensional tensor with
  // four float32 values.
  float input_values[] = {1.0f, 1.0f, 1.0f, 1.0f};
  TensorImpl::SizesType sizes[] = {std::size(input_values)};
  TensorImpl input(ScalarType::Float, std::size(sizes), sizes, input_values);

  // Run the model.
  auto result = module.forward({EValue(Tensor(&input))});
  if (!result.ok()) {
    printf("Inference error: 0x%x\n", (unsigned int)result.error());
    return -1;
  }

  // Retrieve and print the output tensor.
  printf("Output: ");
  auto output = (*result)[0].toTensor();
  for (auto i = 0; i < output.numel(); i++) {
    printf("%f ", output.const_data_ptr<float>()[i]);
  }
  printf("\n");
}
