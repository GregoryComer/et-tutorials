#!/bin/bash

set -euo pipefail

readonly VOCAB_URL="https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json"
readonly VOCAB_JSON="/tmp/nanogpt-vocab.json"
readonly VOCAB_DATA="/tmp/nanogpt-vocab.data"

if [[ ! -f "${VOCAB_JSON}" ]]; then
  echo "Downloading ${VOCAB_JSON}"
  curl "${VOCAB_URL}" > "${VOCAB_JSON}"
else
  echo "${VOCAB_JSON} already downloaded"
fi

set -x # Print the remaining commands as they run

./parse_vocab.py < "${VOCAB_JSON}" > "${VOCAB_DATA}"

c++ -std=c++17 test.cpp -o /tmp/test-tokenizer

/tmp/test-tokenizer "${VOCAB_DATA}" "$@"

