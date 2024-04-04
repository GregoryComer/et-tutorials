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
#include <executorch/sdk/etdump/etdump_flatcc.h>


using namespace std;
using namespace torch::executor;

using SizesType = exec_aten::SizesType;
using DimOrderType = exec_aten::DimOrderType;
using StridesType = exec_aten::StridesType;


vector<int64_t> generate(Module& llm_model, vector<int64_t>& tokens) {
    // Convert the input tokens from a vector of int64_t to EValue.
    // Evalue is a unified data type in the executorch runtime.
    ManagedTensor tensor_tokens(tokens.data(), {1, 3}, ScalarType::Long);
    vector<EValue> inputs = {tensor_tokens.get_aliasing_tensor()};

    // Run the model given the Evalue inputs. The model will also return a sequence of EValues as output.
    Result<vector<EValue>> output_res = llm_model.forward(inputs);

    // Convert the output from EValue to a vector of int64_t.
    Tensor output_tensor = output_res.get()[0].toTensor();
    vector<int64_t> output_data(output_tensor.data_ptr<int64_t>(), output_tensor.data_ptr<int64_t>() + output_tensor.numel());

    return output_data;
}


int main() {
    // Load the input. Here we use "Hello world!" as sample input
    string prompt = "Hello world!";
    cout << "prompt: " << prompt << endl;

    // Load tokenizer and model.
    // The tokenizer is used to tokenize the input and decode the output.
    // The model is the exported nanoGPT model used to generate the output.
    BasicTokenizer tokenizer("vocab.json");

    std::unique_ptr<torch::executor::ETDumpGen> etdump_gen_ =
      std::make_unique<torch::executor::ETDumpGen>();
    Module llm_model("nanogpt.pte", Module::MlockConfig::UseMlock, std::move(etdump_gen_));

    // Tokenize the input. This means we're converting the input text into a sequence of tokens,
    // which is a format that our model can understand. Each token represents a word or a part of a word.
    vector<int64_t> tokens = tokenizer.encode(prompt);

    // Generate outputs. This is where our model is used to process the tokenized input.
    // The model will return a sequence of tokens as output.
    vector<int64_t> outputs = generate(llm_model, tokens);

    torch::executor::ETDumpGen* etdump_gen =
        static_cast<torch::executor::ETDumpGen*>(llm_model.event_tracer());

    ET_LOG(Info, "ETDump size: %zu blocks", etdump_gen->get_num_blocks());
    etdump_result result = etdump_gen->get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
        // On a device with a file system users can just write it out
        // to the file-system.
        FILE* f = fopen("etdump.etdp", "w+");
        fwrite((uint8_t*)result.buf, 1, result.size, f);
        fclose(f);
        free(result.buf);
    }

    // Decode the output. This means we're converting the sequence of tokens back into human-readable text.
    // This is the text that our model generated based on the input.
    string out_str = tokenizer.decode(outputs);

    // Print the generated text.
    cout << "output: " <<  out_str << endl;
}
