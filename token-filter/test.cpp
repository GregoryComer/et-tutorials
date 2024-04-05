#include "tokenizer.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Missing file argument" << std::endl;
    return 1;
  }
  std::optional<BasicTokenizer> tokenizer = BasicTokenizer::load(argv[1]);
  if (!tokenizer.has_value()) {
    return 1;
  }

  std::string prompt = "Hello World!";
  if (argc > 2) {
    prompt = argv[2];
  }
  std::cout << "Using prompt \"" << prompt << "\"" << std::endl;

  std::vector<int64_t> tokens = tokenizer->encode(prompt);
  std::cout << "Token values:" << std::endl;
  for (auto token : tokens) {
    std::cout << "  " << token << std::endl;
  }

  std::string decoded_prompt = tokenizer->decode(tokens);
  std::cout << "Decoded prompt: " << decoded_prompt << std::endl;

  return 0;
}
