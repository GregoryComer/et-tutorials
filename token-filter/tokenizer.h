#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class BasicTokenizer {
public:
    static std::optional<BasicTokenizer> load(const std::string& file_path) {
        // `ate` opens at the end of the file so we can get the size.
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "Unable to open file " << file_path << std::endl;
            return std::nullopt;
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read the entire file.
        std::vector<uint8_t> buffer(size);
        if (!file.read((char*)buffer.data(), size)) {
            std::cerr << "Unable to read " << size
                      << " bytes from file " << file_path << std::endl;
            return std::nullopt;
        }

        // Mappings between words and token values.
        std::unordered_map<std::string, int64_t> encode_map;
        std::unordered_map<int64_t, std::string> decode_map;

        // Parse records from the file.
        uint8_t* p = buffer.data();
        const uint8_t* end = p + size;
        while (p < end) {
            // Little-endian uint32_t: number of UTF-8 bytes
            if (p + 4 > end) {
              std::cerr << "Unexpected EOF" << std::endl;
              return std::nullopt;
            }
            uint32_t nbytes = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24;
            p += 4;

            // UTF-8 bytes of the word.
            if (p + nbytes > end) {
              std::cerr << "Unexpected EOF" << std::endl;
              return std::nullopt;
            }
            std::string word((char*)p, nbytes);
            p += nbytes;

            // Little-endian uint32_t: token value
            if (p + 4 > end) {
              std::cerr << "Unexpected EOF" << std::endl;
              return std::nullopt;
            }
            uint32_t token_value = p[0] | p[1] << 8 | p[2] << 16 | p[3] << 24;
            p += 4;

            // Add to the maps.
            encode_map[word] = token_value;
            decode_map[token_value] = word;
        }

        return BasicTokenizer(std::move(encode_map), std::move(decode_map));
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
    BasicTokenizer(
        std::unordered_map<std::string, int64_t>&& encode,
        std::unordered_map<int64_t, std::string>&& decode):
      encode_(encode), decode_(decode) {}

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

    std::unordered_map<std::string, int64_t> encode_;
    std::unordered_map<int64_t, std::string> decode_;
};
