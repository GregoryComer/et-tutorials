#include <executorch/extension/module/module.h>

#include <cstdio>

using namespace ::torch::executor;

const std::string ModelFile = "SimpleModel.pte";

int main() {
  // Load the model .pte file created by the export process. The Module type
  // encapsulates the loaded model.
  printf("Loading executorch program...\n");
    Module module(ModelFile.c_str());
    
  // Set up the model input data. This model takes a 1-dimensions tensor with
  // four float32 values.
    float input_values[4] = { 1.0f, 1.0f, 1.0f,1.0f };
    int32_t sizes[] = {4};
    TensorImpl input(ScalarType::Float, std::size(sizes), sizes, input_values);

   // Run the model.
    auto result = module.forward({EValue(Tensor(&input))});

    if (!result.ok()) {
        printf("Inference error: %i\n", result.error());
    }

   // Retrieve and print the output tensor.
    printf("Output: ");
    auto output = (*result)[0].toTensor();
    for (auto i = 0; i < output.numel(); i++) {
        printf("%f ", output.const_data_ptr<float>()[i]);
    }
    printf("\n");
}
