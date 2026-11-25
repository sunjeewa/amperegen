#!/usr/bin/env python3
"""Thin Unix-socket tokenizer & chat-template sidecar for amperegen.

Uses the exact HuggingFace AutoTokenizer + chat_template.jinja from the checkpoint
so tokenization, chat formatting (enable_thinking, tools), and incremental UTF-8
detokenization are 100% byte-identical to SGLang, completely off the per-token GPU path.
"""
import argparse
import json
import os
import signal
import socket
import socketserver
import sys
from transformers import AutoTokenizer


class TokenizerService:
    def __init__(self, model_dir: str):
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    @staticmethod
    def _normalize_messages(raw_messages: list) -> list:
        """Normalize OpenAI Chat Completions messages for Qwen3.8 chat_template.jinja.

        - Converts `developer` role -> `system` and merges multiple system messages at index 0.
        - Parses assistant `tool_calls[].function.arguments` JSON strings into dicts so
          Jinja's `tool_call.arguments|items` renders multi-turn tool history cleanly.
        - Copies `reasoning` -> `reasoning_content` fallback if needed.
        - Normalizes non-string/non-list content values (e.g. dicts/ints) into JSON strings.
        """
        system_parts = []
        other_msgs = []
        for orig in raw_messages:
            if not isinstance(orig, dict):
                continue
            msg = dict(orig)
            role = msg.get("role", "")

            # Normalize content type if unexpected
            content = msg.get("content")
            if content is not None and not isinstance(content, (str, list)):
                msg["content"] = json.dumps(content, ensure_ascii=False)

            if role in ("system", "developer"):
                if isinstance(content, str):
                    if content.strip():
                        system_parts.append(content.strip())
                elif isinstance(content, list):
                    texts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and isinstance(p.get("text"), str)
                    ]
                    joined = "\n".join(t for t in texts if t).strip()
                    if joined:
                        system_parts.append(joined)
                continue

            if role == "assistant":
                if not msg.get("reasoning_content") and isinstance(msg.get("reasoning"), str):
                    msg["reasoning_content"] = msg["reasoning"]

                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    norm_tcs = []
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        tc_copy = dict(tc)
                        fn = tc_copy.get("function")
                        if isinstance(fn, dict):
                            fn_copy = dict(fn)
                            args = fn_copy.get("arguments")
                            if isinstance(args, str):
                                s = args.strip()
                                if not s:
                                    fn_copy["arguments"] = {}
                                else:
                                    try:
                                        parsed = json.loads(s)
                                        fn_copy["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
                                    except Exception:
                                        fn_copy["arguments"] = {"raw": s}
                            elif args is None:
                                fn_copy["arguments"] = {}
                            tc_copy["function"] = fn_copy
                        elif isinstance(tc_copy.get("arguments"), str):
                            s = tc_copy["arguments"].strip()
                            if not s:
                                tc_copy["arguments"] = {}
                            else:
                                try:
                                    parsed = json.loads(s)
                                    tc_copy["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
                                except Exception:
                                    tc_copy["arguments"] = {"raw": s}
                        norm_tcs.append(tc_copy)
                    msg["tool_calls"] = norm_tcs

            other_msgs.append(msg)

        out = []
        if system_parts:
            out.append({"role": "system", "content": "\n\n".join(system_parts)})
        out.extend(other_msgs)
        return out

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True}
        if op == "tokenize":
            text = req.get("text", "")
            add_special = bool(req.get("add_special_tokens", False))
            ids = self.tok.encode(text, add_special_tokens=add_special)
            return {"token_ids": ids}
        if op == "render_chat":
            messages = self._normalize_messages(req.get("messages", []))
            tools = req.get("tools")
            kwargs = req.get("chat_template_kwargs") or {}
            add_gen = bool(req.get("add_generation_prompt", True))
            tpl_kwargs = dict(kwargs)
            # Qwen3.8 chat_template.jinja only accepts reasoning_effort in ('xhigh','high','medium','low').
            # Map OpenAI's reasoning_effort="none" -> enable_thinking=False.
            re_val = tpl_kwargs.get("reasoning_effort")
            if isinstance(re_val, str) and re_val.lower() in ("none", "off", "disabled"):
                tpl_kwargs.pop("reasoning_effort", None)
                tpl_kwargs.setdefault("enable_thinking", False)
            if tools is not None:
                tpl_kwargs["tools"] = tools
            rendered = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_gen,
                **tpl_kwargs,
            )
            ids = self.tok.encode(rendered, add_special_tokens=False)
            in_think = rendered.rstrip().endswith("<think>")
            return {"token_ids": ids, "in_think": in_think}
        if op == "detokenize":
            ids = req.get("token_ids", [])
            skip_special = bool(req.get("skip_special_tokens", True))
            text = self.tok.decode(ids, skip_special_tokens=skip_special)
            return {"text": text}
        return {"error": f"unknown op: {op}"}


def make_handler(svc: TokenizerService):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                try:
                    req = json.loads(line.decode("utf-8"))
                    resp = svc.handle(req)
                except Exception as e:
                    resp = {"error": str(e)}
                out = (json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8")
                self.wfile.write(out)
                self.wfile.flush()
    return Handler


class ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--socket", required=True)
    args = ap.parse_args()

    if os.path.exists(args.socket):
        os.unlink(args.socket)

    svc = TokenizerService(args.model_dir)
    srv = ThreadingUnixServer(args.socket, make_handler(svc))

    def shutdown_handler(_signum, _frame):
        try:
            if os.path.exists(args.socket):
                os.unlink(args.socket)
        except OSError:
            pass
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    sys.stderr.write(f"[sidecar] Tokenizer ready on {args.socket}\n")
    sys.stderr.flush()
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
