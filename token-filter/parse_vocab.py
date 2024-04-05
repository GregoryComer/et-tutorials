#!/usr/bin/env python

import json
import re
import struct
import sys

# Parse JSON data form standard input.
vocab_json = json.load(sys.stdin)

# Write an easy-to-parse binary format to stdout.
outfp = sys.stdout.buffer
for word, token_str in vocab_json.items():
    # The JSON encoding uses placeholders for some characters.
    word = re.sub("\u0120", " ", word)
    word = re.sub("\u010a", "\n", word)

    word_bytes = word.encode("UTF-8")

    # Write an entry to the file:
    # - 4 bytes: Value of the token associated with the word
    outfp.write(struct.pack("<I", len(word_bytes)))
    # - N bytes: The word as UTF-8 bytes
    outfp.write(word_bytes)
    # - 4 bytes: Length of the word (little-endian uint32_t)
    outfp.write(struct.pack("<I", int(token_str)))
