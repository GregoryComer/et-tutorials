#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>

class BasicTokenizer {
public:
    BasicTokenizer(const std::string& filePath) {
        std::ifstream file(filePath);
        std::string line;
        int64_t index = 0;
        while (std::getline(file, line)) {
            // Skip lines with only a single brace
            if (line[0] == '{' or line[0] == '}') {
                continue;
            }

            size_t split_pos = line.find("\": ");
            if (split_pos == std::string::npos) {
                continue;
            }
            std::string key = line.substr(5, split_pos-5); // 5 here to remove the starting spaces
            int64_t value = std::stoi(line.substr(split_pos + 3, line.size() - split_pos - 4)); // -1 here to remove the ending comma

            // remove duplicate backslashes
            // TODO: adopt byte encoder to handle unicode characters in json file.
            for (size_t idx = 0; idx < key.length(); idx++) {
                if (key[idx] == '\\') {
                    key.erase(idx, 1);
                    if (key[idx] == '\\') {
                        // If there are two backslashes, keep the second one
                        idx += 1;
                    }
                }
            }

            dictionary[key] = value;
            reverse_dictionary[value] = key;
        }
    }
    std::vector<int64_t> encode(const std::string& prompt) {
        std::vector<int64_t> result;
        std::istringstream iss(prompt);
        for(std::string s; iss >> s; ) {
            result.push_back(dictionary[s]);
        }
        return result;
    }
    std::string decode(const std::vector<int64_t>& indices) {
        std::string result;
        for (const auto& index : indices) {
            result += reverse_dictionary[index] + " ";
        }
        return result;
    }
private:
    std::unordered_map<std::string, int64_t> dictionary;
    std::unordered_map<int64_t, std::string> reverse_dictionary;
};
