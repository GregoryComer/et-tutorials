#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <nlohmann/json.hpp>
class BasicTokenizer {
public:
    BasicTokenizer(const std::string& filePath) {
        std::ifstream i(filePath);
        nlohmann::json j;
        i >> j;
        for (auto& element : j.items()) {
            dictionary[element.key()] = element.value();
            reverse_dictionary[element.value()] = element.key();
        }
    }
    std::vector<int> encode(const std::string& prompt) {
        std::vector<int> result;
        std::istringstream iss(prompt);
        for(std::string s; iss >> s; ) {
            result.push_back(dictionary[s]);
        }
        return result;
    }
    std::string decode(const std::vector<int>& indices) {
        std::string result;
        for (const auto& index : indices) {
            result += reverse_dictionary[index] + " ";
        }
        return result;
    }
private:
    std::unordered_map<std::string, int> dictionary;
    std::unordered_map<int, std::string> reverse_dictionary;
};
