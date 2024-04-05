/*
The simple runner for nanoGPT. It is for demonstrating how to run the model exported to executorch runtime.

*/

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>

#include "basic_tokenizer.h"
#include "basic_sampler.h"
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


vector<int64_t> generate(Module& llm_model, vector<int64_t>& input_tokens, BasicSampler& sampler, size_t target_output_length) {
    vector<int64_t> output_tokens;

    for (int i = 0; i < target_output_length; i++) {
        // Convert the input_tokens from a vector of int64_t to EValue.
        // Evalue is a unified data type in the executorch runtime.
        ManagedTensor tensor_tokens(input_tokens.data(), {1, 8}, ScalarType::Long);
        vector<EValue> inputs = {tensor_tokens.get_tensor()};

        // Run the model given the Evalue inputs. The model will also return a sequence of EValues as output.
        Result<vector<EValue>> logits_evalue = llm_model.forward(inputs);

        // Convert the output from EValue to a logits in float.
        Tensor logits_tensor = logits_evalue.get()[0].toTensor();
        vector<float> logits(logits_tensor.data_ptr<float>(), logits_tensor.data_ptr<float>() + logits_tensor.numel());

        // Sample the next token from the logits.
        int64_t next_token = sampler.sample(logits);

        // Record the next token
        output_tokens.push_back(next_token);

        // Update next input.
        input_tokens.erase(input_tokens.begin());
        input_tokens.push_back(next_token);
    }

    return output_tokens;
}


int main() {
    // Load the input.
    string prompt = "Hello world, nice to see you!";
    cout << "prompt: " << prompt << endl;

    // Load tokenizer.
    // The tokenizer is used to tokenize the input and decode the output.
    BasicTokenizer tokenizer("vocab.json");
    BasicSampler sampler = BasicSampler();

    // Load exported nanoGPT model, which will be used to generate the output.
    Module llm_model("nanogpt.pte");

    // Convert the input text into a list of integers (tokens) that represents it, using the string-to-token
    // mapping that the model was trained on. Each token is an integer that represents a word or part of a word.
    vector<int64_t> tokens = tokenizer.encode(prompt);

    // Generate outputs. This is where our model is used to process the tokenized input.
    // The model will return a sequence of tokens as output.
    vector<int64_t> outputs = generate(llm_model, tokens, sampler, /*target_output_length*/20);

    // Decode the output. This means we're converting the sequence of tokens back into human-readable text.
    // This is the text that our model generated based on the input.
    string out_str = tokenizer.decode(outputs);

    // Print the generated text.
    cout << "output: " <<  out_str << endl;
}
