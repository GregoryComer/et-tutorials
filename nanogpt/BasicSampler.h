#include <vector>
#include <algorithm>
class Sampler {
public:
    // Constructor
    Sampler() {}
    // Sample function
    int sample(std::vector<double> logits) {
        // Find the maximum element's index
        int max_index = std::max_element(logits.begin(), logits.end()) - logits.begin();
        return max_index;
    }
};
