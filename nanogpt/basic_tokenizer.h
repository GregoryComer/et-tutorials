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

        if (!file) {
            std::cerr << "Unable to open file";
            exit(9); // return with error code
        }
        std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        // Remove the first and last braces
        str[0] = ' ';
        str.pop_back();

        std::stringstream ss(str);
        std::string kv_pair;
        while (std::getline(ss, kv_pair, ',')) {
            size_t split_pos = kv_pair.find("\": ");
            if (split_pos == std::string::npos) {
                continue;
            }
            std::string key = kv_pair.substr(2, split_pos-2); // 5 here to remove the starting spaces
            int64_t value = std::stoi(kv_pair.substr(split_pos + 3, kv_pair.size() - split_pos - 3)); // -4 here to remove the ending comma

            key = post_process_key(key);

            encode_[key] = value;
            decode_[value] = key;
        }
    }
    std::vector<int64_t> encode(const std::string& prompt) {
        std::vector<std::string> words = parse_prompt(prompt);
        std::vector<int64_t> result;
        for (auto word: words) {
            result.push_back(encode_[word]);
        }
        return result;
    }
    std::string decode(const std::vector<int64_t>& indices) {
        std::string result;
        for (const auto& index : indices) {
            result += decode_[index];
        }
        return result;
    }
private:
    std::unordered_map<std::string, int64_t> encode_;
    std::unordered_map<int64_t, std::string> decode_;

    std::string post_process_key(std::string key) {
        // Replace the unicode characters with the corresponding byte encoding
        // TODO: adopt byte encoder to handle unicode characters in json file.

        std::unordered_map<std::string, std::string> replacements = {
            {"\\u0120", " "},
            {"\\u010a", "\n"},
        };

        for (const auto& replacement : replacements) {
            size_t pos = 0;
            // While loop through all instances of the substring in the string
            while((pos = key.find(replacement.first, pos)) != std::string::npos) {
                key.replace(pos, replacement.first.length(), replacement.second);
                pos += replacement.second.length();
            }
        }


        // remove duplicate backslashes
        for (size_t idx = 0; idx < key.length(); idx++) {
            if (key[idx] == '\\') {
                key.erase(idx, 1);
                if (key[idx] == '\\') {
                    // If there are two backslashes, keep the second one
                    idx += 1;
                }
            }
        }

        return key;
    }
    std::vector<std::string> parse_prompt(const std::string& prompt) {
        std::vector<std::string> result;
        std::string word;
        for (char c : prompt) {
            if (c == ' ') {
                if (!word.empty()) {
                    result.push_back(word);
                    word.clear();
                }
                word += c;
            } else if (ispunct(c)) {
                if (!word.empty()) {
                    result.push_back(word);
                    word.clear();
                }
                result.push_back(std::string(1, c));
            } else {
                word += c;
            }
        }
        if (!word.empty()) {
            result.push_back(word);
        }
        return result;
    }
};
