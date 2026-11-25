#include "server/tool_parser.hpp"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace amperegen;

namespace {

void check_stream_equals_nonstream(const std::string& text, const Json* tools, const std::string& label) {
    auto ns = parse_qwen3_coder_non_stream(text, tools, "call_test_");

    for (size_t step : {size_t(1), size_t(2), size_t(3), size_t(5), size_t(11), size_t(37)}) {
        Qwen3CoderStreamParser p(tools, "call_test_");
        std::string streamed_text;
        std::vector<ParsedToolCall> streamed_calls;

        auto apply = [&](const StreamParseOutput& o) {
            streamed_text += o.normal_text;
            for (const auto& d : o.tool_deltas) {
                if (d.is_header) {
                    assert((int)streamed_calls.size() == d.index);
                    streamed_calls.push_back(ParsedToolCall{d.index, d.id, d.name, d.arguments_delta});
                } else {
                    assert(d.index >= 0 && d.index < (int)streamed_calls.size());
                    streamed_calls[d.index].arguments += d.arguments_delta;
                }
            }
        };

        for (size_t i = 0; i < text.size(); i += step) {
            apply(p.feed(std::string_view(text).substr(i, std::min(step, text.size() - i))));
        }
        apply(p.finish());

        std::string st_trim = streamed_text;
        if (!streamed_calls.empty()) {
            while (!st_trim.empty() && std::isspace((unsigned char)st_trim.back())) st_trim.pop_back();
        }
        assert(st_trim == ns.normal_text);
        assert(streamed_calls.size() == ns.tool_calls.size());
        for (size_t i = 0; i < streamed_calls.size(); ++i) {
            assert(streamed_calls[i].id == ns.tool_calls[i].id);
            assert(streamed_calls[i].name == ns.tool_calls[i].name);
            assert(streamed_calls[i].arguments == ns.tool_calls[i].arguments);
            Json parsed_args = Json::parse(streamed_calls[i].arguments);
            assert(parsed_args.is_object());
        }
    }
    printf("[PASS] %s (calls=%zu, text=%zu bytes)\n", label.c_str(), ns.tool_calls.size(), ns.normal_text.size());
}

} // namespace

int main() {
    Json tools = Json::parse(R"([
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "cwd": {"type": ["string", "null"]}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "raw": {"type": "boolean"},
                        "scale": {"type": "number"},
                        "tags": {"type": "array"},
                        "meta": {"type": "object"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "noop",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ])");

    // 1. Single shell tool call with preceding prose
    {
        std::string s =
            "Let's inspect the directory layout and build files.\n\n"
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>\n"
            "ls -la && find . -maxdepth 2 -type f | head -50\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>";
        check_stream_equals_nonstream(s, &tools, "single_bash_call");
        auto ns = parse_qwen3_coder_non_stream(s, &tools, "call_test_");
        assert(ns.normal_text == "Let's inspect the directory layout and build files.");
        assert(ns.tool_calls.size() == 1);
        assert(ns.tool_calls[0].name == "bash");
        Json args = Json::parse(ns.tool_calls[0].arguments);
        assert(args.at("command").as_string() == "ls -la && find . -maxdepth 2 -type f | head -50");
    }

    // 2. Parallel tool calls + typed params (int, bool, float, array, object, null, multiline code with < >)
    {
        std::string s =
            "<tool_call>\n"
            "<function=read>\n"
            "<parameter=path>\n"
            "src/main.cpp\n"
            "</parameter>\n"
            "<parameter=offset>\n"
            "42\n"
            "</parameter>\n"
            "<parameter=raw>\n"
            "true\n"
            "</parameter>\n"
            "<parameter=scale>\n"
            "1.5\n"
            "</parameter>\n"
            "<parameter=tags>\n"
            "[\"a\", \"b\"]\n"
            "</parameter>\n"
            "<parameter=meta>\n"
            "{\"k\": 123}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=bash>\n"
            "<parameter=command>\n"
            "cat <<'EOF' > /tmp/x.cpp\n"
            "#include <vector>\n"
            "int main() { return 1 < 2 ? 0 : 1; }\n"
            "EOF\n"
            "</parameter>\n"
            "<parameter=timeout>\n"
            "30\n"
            "</parameter>\n"
            "<parameter=cwd>\n"
            "null\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=noop>\n"
            "</function>\n"
            "</tool_call>";
        check_stream_equals_nonstream(s, &tools, "parallel_typed_multiline");
        auto ns = parse_qwen3_coder_non_stream(s, &tools, "call_test_");
        assert(ns.normal_text.empty());
        assert(ns.tool_calls.size() == 3);
        Json a0 = Json::parse(ns.tool_calls[0].arguments);
        assert(a0.at("path").as_string() == "src/main.cpp");
        assert(a0.at("offset").as_int() == 42);
        assert(a0.at("raw").as_bool() == true);
        assert(a0.at("scale").as_double() == 1.5);
        assert(a0.at("tags").as_array().size() == 2);
        assert(a0.at("meta").at("k").as_int() == 123);

        Json a1 = Json::parse(ns.tool_calls[1].arguments);
        assert(a1.at("timeout").as_int() == 30);
        assert(a1.at("cwd").is_null());
        assert(a1.at("command").as_string().find("#include <vector>") != std::string::npos);

        assert(ns.tool_calls[2].name == "noop");
        assert(ns.tool_calls[2].arguments == "{}");
    }

    // 3. Plain text with '<' and partial tag at EOF
    {
        std::string s = "Check if a < b and <div>ok</div> or ends with <tool";
        check_stream_equals_nonstream(s, &tools, "plain_text_angle_brackets");
        auto ns = parse_qwen3_coder_non_stream(s, &tools, "call_test_");
        assert(ns.tool_calls.empty());
        assert(ns.normal_text == s);
    }

    // 4. JSON fallback format inside <tool_call>
    {
        std::string s = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call>";
        check_stream_equals_nonstream(s, &tools, "json_fallback");
        auto ns = parse_qwen3_coder_non_stream(s, &tools, "call_test_");
        assert(ns.tool_calls.size() == 1);
        assert(ns.tool_calls[0].name == "bash");
        assert(Json::parse(ns.tool_calls[0].arguments).at("command").as_string() == "pwd");
    }

    // 5. Truncated mid-parameter (max_tokens cutoff) still produces valid closed JSON
    {
        std::string s = "<tool_call>\n<function=bash>\n<parameter=command>\necho hi";
        check_stream_equals_nonstream(s, &tools, "truncated_cutoff");
        auto ns = parse_qwen3_coder_non_stream(s, &tools, "call_test_");
        assert(ns.tool_calls.size() == 1);
        assert(Json::parse(ns.tool_calls[0].arguments).at("command").as_string() == "echo hi");
    }

    printf("ALL UNIT TESTS PASSED!\n");
    return 0;
}
