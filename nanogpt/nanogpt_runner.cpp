#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "basic_tokenizer.h"
#include "basic_sampler.h"
#include "managed_tensor.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/evalue_util/print_evalue.h>


using namespace std;
using namespace torch::executor;


const int maxSeqLen = 1024;


vector<int64_t> evalue_vector_to_int_vector(vector<EValue>& evalue) {
    vector<int64_t> int_vector(evalue.size());
    for (int i = 0; i < evalue.size(); i++) {
        int_vector.push_back(evalue[i].toInt());
    }
    return int_vector;
}

// TODO: should convert to evalue tensor instead of evalue int
vector<EValue> int_vector_to_evalue_vector(vector<int64_t>& int_vector) {
    vector<EValue> evalue_vector(int_vector.size());
    for (int i = 0; i < int_vector.size(); i++) {
        evalue_vector.push_back(EValue(int_vector[i]));
    }
    return evalue_vector;
}

Result<vector<int64_t>> generate(std::unique_ptr<Module>& llm_model, Sampler sampler, vector<int64_t>& tokens, int64_t max_length) {
    // block size will limit the input size; check nanogpt.config.block_size for more details.

    // std::vector<EValue> inputs = int_vector_to_evalue_vector(tokens);
    // int64_t input_length = tokens.size();

    // // placeholder
    // int64_t eos_token = 50256; // eos token in gpt2
    // int64_t cur_token = 0;


    // bool is_done = false;

    // for (int64_t pos = 0; pos < max_length && (!is_done); pos++) {
    //     Result<std::vector<EValue>> logits_res = llm_model->forward(inputs);
    //     ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());

    //     cur_token = sampler.sample(logits_res.get(), pos);
    //     inputs.push_back(cur_token);

    //     if (cur_token == eos_token) {
    //         is_done = true;
    //     }
    // }

    // std::vector<EValue> outputs(inputs.begin() + input_length, inputs.end());
    // std::vector<int64_t> outputs_tokens = evalue_vector_to_int_vector(outputs);

    ManagedTensor tensor_tokens(
      tokens.data(),
      {1, 3},
      ScalarType::Long);

    // tensor_tokens.resize({1, 1});

    cout << "111111111111" << endl;

    std::vector<EValue> inputs;
    inputs.push_back(tensor_tokens.get_aliasing_tensor());

    cout << inputs.size() << endl;
    cout << inputs[0].toTensor().data_ptr<int64_t>()[0] << endl;

    Result<std::vector<EValue>> output_res = llm_model->forward(inputs);

    cout << "222222222222" << endl;
    cout << output_res.get().size() << endl;

    Tensor output_tensor = output_res.get()[0].toTensor();

    vector<int64_t> output_data(output_tensor.data_ptr<int64_t>(), output_tensor.data_ptr<int64_t>() + output_tensor.numel());

    cout << "333333333333" << endl;
    cout << output_tensor.numel() << endl;

    // for (auto od: output_data) {
    //     cout << od << ", ";
    // }

    // cout << endl;

    // Return the logits tensor
    return output_data;
}


int main() {
    // 1. get input
    string prompt = "I";

    // 2. load tokenizer, model and sampler.
    BasicTokenizer tokenizer("local_vocab.json");

    std::unique_ptr<Module> llm_model = std::make_unique<Module>(
          /*model_path=*/"nanogpt.pte",
          Module::MlockConfig::UseMlockIgnoreErrors);
    Sampler sampler = Sampler(/*vocab_size=*/maxSeqLen);

    // 3. tokenize the input
    // vector<int64_t> tokens = tokenizer.encode(prompt);
    vector<int64_t> tokens = {15496, 995, 0};

    // 4. generate outputs
    Result<vector<int64_t>> outputs = generate(llm_model, sampler, tokens, /*max_length=*/10);

    // 5. decode the outputs
    string out_str = tokenizer.decode(outputs.get());

    // 6. print the outputs
    cout << out_str;

}
