#include <vector>
#include <algorithm>
#include <executorch/extension/evalue_util/print_evalue.h>

using namespace torch::executor;

class Sampler {
public:
    // Constructor
    Sampler(int64_t vocab_size) {
        vocab_size_ = vocab_size;
    }
    // Sample function
    int sample(std::vector<EValue> logits, int64_t pos) {
        // Find the maximum element's index
        float* logits_float = logits[0].toTensor().mutable_data_ptr<float>() + pos * vocab_size_;

        int max_index = std::max_element(logits_float, logits_float + vocab_size_) - logits_float;
        return max_index;
    }

private:
    // Vocabulary size
    int64_t vocab_size_;
};
