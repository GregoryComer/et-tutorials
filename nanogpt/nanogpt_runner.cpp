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

const int maxSeqLen = 1024;

Result<vector<int64_t>> generate(unique_ptr<Module>& llm_model, vector<int64_t>& tokens) {
    ManagedTensor tensor_tokens(tokens.data(), {1, 3}, ScalarType::Long);

    vector<EValue> inputs;
    inputs.push_back(tensor_tokens.get_aliasing_tensor());

    Result<vector<EValue>> output_res = llm_model->forward(inputs);

    Tensor output_tensor = output_res.get()[0].toTensor();
    vector<int64_t> output_data(output_tensor.data_ptr<int64_t>(), output_tensor.data_ptr<int64_t>() + output_tensor.numel());

    return output_data;
}


int main() {
    // 1. get input
    string prompt = "Hello world!";
    cout << "prompt: " << prompt << endl;

    // 2. load tokenizer, model and sampler.
    BasicTokenizer tokenizer("local_vocab.json");

    std::unique_ptr<torch::executor::ETDumpGen> etdump_gen_ =
      std::make_unique<torch::executor::ETDumpGen>();

    unique_ptr<Module> llm_model = make_unique<Module>(
          /*model_path=*/"nanogpt.pte",
          Module::MlockConfig::UseMlockIgnoreErrors,
          std::move(etdump_gen_));

    // 3. tokenize the input
    vector<int64_t> tokens = tokenizer.encode(prompt);

    // 4. generate outputs
    Result<vector<int64_t>> outputs = generate(llm_model, tokens);

    // Write ETDump to file
    torch::executor::ETDumpGen* etdump_gen =
        static_cast<torch::executor::ETDumpGen*>(llm_model->event_tracer());

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

    // 5. decode the outputs
    string out_str = tokenizer.decode(outputs.get());

    // 6. print the outputs
    cout << "output: " <<  out_str << endl;

}
