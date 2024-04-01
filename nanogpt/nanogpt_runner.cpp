#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "BasicTokenizer.h"
#include "BasicSampler.h"

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>


using namespace std;

vector<int64_t>& evalue_vector_to_int_vector(vector<EValue>& evalue) {
    vector<int64_t> int_vector(evalue.size());
    for (int i = 0; i < evalue.size(); i++) {
        int_vector.push_back(evalue[i].toInt());
    }
    return int_vector;
}

// TODO: should convert to evalue tensor instead of evalue int
vector<EValue>& int_vector_to_evalue_vector(vector<int64_t>& int_vector) {
    vector<EValue> evalue_vector(int_vector.size());
    for (int i = 0; i < int_vector.size(); i++) {
        evalue_vector.push_back(EValue(int_vector[i]));
    }
    return evalue_vector;
}

vector<int64_t> generate(std::unique_ptr<Module> llm_model, Sampler sampler, vector<int64_t>& tokens, int64_t max_length) {
    // block size will limit the input size; check nanogpt.config.block_size for more details.

    std::vector<EValue> inputs = int_vector_to_evalue_vector(tokens);
    int64_t input_length = tokens.size();

    // placeholder
    int64_t eos_token = 0;

    std::vector<EValue> outputs;

    bool is_done = false;

    for (int64_t _ = 0; _ < max_length && (!is_done); _++) {
        Result<std::vector<EValue>> logits_res = llm_model->forward(inputs);
        ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());

        cur_token = sampler.sample(logits_res.get());
        inputs.append(cur_token);

        if (cur_token == eos_token) {
            is_done = true;
        }
    }

    std::vector<EValue> outputs(inputs.begin() + input_length, inputs.end());
    std::vector<int64_t> outputs_tokens = evalue_vector_to_int_vector(outputs);

    // Return the logits tensor
    return outputs_tokens;
}


int main() {
    // 1. get input
    string prompt = "an example prompt";

    // 2. load tokenizer, model and sampler.
    BasicTokenizer tokenizer();
    std::unique_ptr<Module> llm_model = std::make_unique<Module>(
          /*model_path=*/"model.pte",
          Module::MlockConfig::UseMlockIgnoreErrors);
    Sampler sampler = Sampler();

    // 3. tokenize the input
    vector<int64_t> tokens = tokenizer.encode(prompt);

    // 4. generate outputs
    vector<int64_t> outputs = generate(llm_model, sampler, tokens);

    // 5. decode the outputs
    string out_str = tokenizer.decode(outputs);

    // 6. print the outputs
    cout << out_str;

}
