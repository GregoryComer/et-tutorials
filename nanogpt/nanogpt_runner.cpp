/*
The simple runner for nanoGPT. It is for demonstrating how to run the model exported to executorch runtime.

*/

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>

#include "basic_tokenizer.h"
#include "managed_tensor.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using namespace std;
using namespace torch::executor;

using SizesType = exec_aten::SizesType;
using DimOrderType = exec_aten::DimOrderType;
using StridesType = exec_aten::StridesType;


vector<int64_t> generate(Module& llm_model, vector<int64_t>& tokens) {
    // Convert the input tokens from a vector of int64_t to EValue.
    // Evalue is a unified data type in the executorch runtime.
    ManagedTensor tensor_tokens(tokens.data(), {1, 3}, ScalarType::Long);
    vector<EValue> inputs = {tensor_tokens.get_tensor()};

    // Run the model given the Evalue inputs. The model will also return a sequence of EValues as output.
    Result<vector<EValue>> output_res = llm_model.forward(inputs);

    // Convert the output from EValue to a vector of int64_t.
    Tensor output_tensor = output_res.get()[0].toTensor();
    vector<int64_t> output_data(output_tensor.data_ptr<int64_t>(), output_tensor.data_ptr<int64_t>() + output_tensor.numel());

    return output_data;
}


int main() {
    // Load the input.
    string prompt = "Hello world!";
    cout << "prompt: " << prompt << endl;

    // Load tokenizer.
    // The tokenizer is used to tokenize the input and decode the output.
    BasicTokenizer tokenizer("vocab.json");

    // Load exported nanoGPT model, which will be used to generate the output.
    Module llm_model("nanogpt.pte");

    // Convert the input text into a list of integers (tokens) that represents it, using the string-to-token
    // mapping that the model was trained on. Each token is an integer that represents a word or part of a word.
    vector<int64_t> tokens = tokenizer.encode(prompt);

    // Generate outputs. This is where our model is used to process the tokenized input.
    // The model will return a sequence of tokens as output.
    vector<int64_t> outputs = generate(llm_model, tokens);

    // Decode the output. This means we're converting the sequence of tokens back into human-readable text.
    // This is the text that our model generated based on the input.
    string out_str = tokenizer.decode(outputs);

    // Print the generated text.
    cout << "output: " <<  out_str << endl;
}
