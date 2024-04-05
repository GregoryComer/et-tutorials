#include <vector>
#include <algorithm>
class BasicSampler {
public:
    // Constructor
    BasicSampler() {}
    // Sample function
    int64_t sample(std::vector<float> logits) {
        // Find the maximum element's index
        int64_t max_index = std::max_element(logits.begin(), logits.end()) - logits.begin();
        return max_index;
    }
};
