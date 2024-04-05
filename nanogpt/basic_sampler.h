#include <vector>
#include <algorithm>
class BasicSampler {
public:
    BasicSampler() {}
    int64_t sample(std::vector<float> logits) {
        // Find the token with the highest log probability.
        int64_t max_index = std::max_element(logits.begin(), logits.end()) - logits.begin();
        return max_index;
    }
};
