# Third-Party Attribution

## SGLang (Apache-2.0) & Flash Linear Attention (MIT)
Model math reference, Gated DeltaNet recurrence ordering, DFlash2 speculative decoding architecture
(`DFlash2DraftModel`, `DFlashGroupedConv`, `CandidateSelector`), and the FP8 E4M3 exponent-bias
(`2^120`) BF16 dequantization technique were verified against and inspired by SGLang 0.5.20
(Copyright SGLang Team, Apache License 2.0) and Flash Linear Attention (FLA, MIT License).

## minja (MIT)
Vendored under `third_party/minja/` (Copyright 2024 Google LLC, MIT License,
[third_party/minja/LICENSE](third_party/minja/LICENSE)) for native C++ Jinja2 chat-template execution.

## nlohmann/json (MIT)
Vendored under `third_party/nlohmann/` (Copyright Niels Lohmann, MIT License,
[third_party/nlohmann/LICENSE.MIT](third_party/nlohmann/LICENSE.MIT)) for JSON parsing in `tokenizer.cpp` and `minja`.

## PCRE2 (BSD-3-Clause) & utf8proc (MIT)
Linked dynamically at runtime for Unicode regex pre-tokenization (`libpcre2-8`) and Unicode normalization (`libutf8proc`).
