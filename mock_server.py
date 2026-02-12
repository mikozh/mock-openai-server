import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Optional dependency, required for GPT-4 tokenization
try:
    import tiktoken
except Exception:
    tiktoken = None

app = FastAPI(title="Mock OpenAI Streaming Server (tiktoken)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "mock-gpt4"
DEFAULT_TOKENIZER_MODEL = "gpt-4"   # used by tiktoken.encoding_for_model
DEFAULT_ENCODING_NAME = "cl100k_base"


def _now_unix() -> int:
    return int(time.time())


def _make_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _make_cmpl_id() -> str:
    return f"cmpl-{uuid.uuid4().hex[:24]}"


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _last_user_content(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            return json.dumps(c, ensure_ascii=False)
    if messages and isinstance(messages[-1], dict):
        c = messages[-1].get("content", "")
        return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


def _get_tiktoken_encoding(
    tokenizer_model: Optional[str] = None,
    encoding_name: Optional[str] = None,
):
    if tiktoken is None:
        raise RuntimeError(
            "tiktoken is not installed. Install it with: pip install tiktoken"
        )

    if tokenizer_model:
        try:
            return tiktoken.encoding_for_model(tokenizer_model)
        except Exception:
            # Fall back to encoding_name if model is unknown
            pass

    return tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME)


def _tokenize_to_pieces_tiktoken(
    text: str,
    tokenizer_model: Optional[str] = None,
    encoding_name: Optional[str] = None,
) -> List[str]:
    """
    Returns list of *decoded* pieces, one per token id.
    Concatenating the pieces yields the original text.
    """
    enc = _get_tiktoken_encoding(tokenizer_model=tokenizer_model, encoding_name=encoding_name)
    token_ids = enc.encode(text, allowed_special="all")
    return [enc.decode([tid]) for tid in token_ids]


def _count_tokens_tiktoken(
    text: str,
    tokenizer_model: Optional[str] = None,
    encoding_name: Optional[str] = None,
) -> int:
    enc = _get_tiktoken_encoding(tokenizer_model=tokenizer_model, encoding_name=encoding_name)
    return len(enc.encode(text, allowed_special="all"))


async def _stream_chat_completion(
    *,
    request_id: str,
    model: str,
    output_text: str,
    tokens_per_second: float,
    time_to_first_token: float,
    tokenizer_model: str,
    encoding_name: str,
    tokens_per_chunk: int,
):
    created = _now_unix()

    pieces = _tokenize_to_pieces_tiktoken(
        output_text,
        tokenizer_model=tokenizer_model,
        encoding_name=encoding_name,
    )

    # TTFT delay
    if time_to_first_token and time_to_first_token > 0:
        await asyncio.sleep(time_to_first_token)

    # Initial chunk (role)
    yield _sse(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    delay = 0.0
    if tokens_per_second and tokens_per_second > 0:
        delay = 1.0 / float(tokens_per_second)

    step = max(1, int(tokens_per_chunk))
    i = 0
    while i < len(pieces):
        chunk_pieces = pieces[i : i + step]
        i += len(chunk_pieces)
        content_piece = "".join(chunk_pieces)

        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content_piece}, "finish_reason": None}],
            }
        )

        if delay > 0:
            await asyncio.sleep(delay)

    # Finish chunk
    yield _sse(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


async def _stream_text_completion(
    *,
    request_id: str,
    model: str,
    output_text: str,
    tokens_per_second: float,
    time_to_first_token: float,
    tokenizer_model: str,
    encoding_name: str,
    tokens_per_chunk: int,
):
    created = _now_unix()

    pieces = _tokenize_to_pieces_tiktoken(
        output_text,
        tokenizer_model=tokenizer_model,
        encoding_name=encoding_name,
    )

    if time_to_first_token and time_to_first_token > 0:
        await asyncio.sleep(time_to_first_token)

    delay = 0.0
    if tokens_per_second and tokens_per_second > 0:
        delay = 1.0 / float(tokens_per_second)

    step = max(1, int(tokens_per_chunk))
    i = 0
    while i < len(pieces):
        chunk_pieces = pieces[i : i + step]
        i += len(chunk_pieces)
        text_piece = "".join(chunk_pieces)

        yield _sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": text_piece, "finish_reason": None}],
            }
        )

        if delay > 0:
            await asyncio.sleep(delay)

    yield _sse(
        {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": DEFAULT_MODEL, "object": "model", "created": 0, "owned_by": "mock"}
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()

    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))

    # What to "generate"
    output_text = (
        body.get("mock_response")
        or body.get("text")
        or body.get("response")
        or _last_user_content(body.get("messages"))
    )

    # Controls
    tokens_per_second = float(body.get("tokens_per_second", body.get("tps", 15.0)))
    time_to_first_token = float(body.get("time_to_first_token", body.get("ttft", 0.2)))

    # Tokenization controls
    tokenizer_model = body.get("tokenizer_model", DEFAULT_TOKENIZER_MODEL)  # e.g. "gpt-4"
    encoding_name = body.get("encoding_name", DEFAULT_ENCODING_NAME)        # fallback, e.g. "cl100k_base"

    # Stream cadence controls
    tokens_per_chunk = int(body.get("tokens_per_chunk", 1))  # default: 1 true token at a time

    request_id = _make_chat_id()
    created = _now_unix()

    if stream:
        return StreamingResponse(
            _stream_chat_completion(
                request_id=request_id,
                model=model,
                output_text=output_text,
                tokens_per_second=tokens_per_second,
                time_to_first_token=time_to_first_token,
                tokenizer_model=tokenizer_model,
                encoding_name=encoding_name,
                tokens_per_chunk=tokens_per_chunk,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming OpenAI-ish response
    completion_tokens = _count_tokens_tiktoken(output_text, tokenizer_model=tokenizer_model, encoding_name=encoding_name)

    prompt_tokens = 0
    msgs = body.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                prompt_tokens += _count_tokens_tiktoken(
                    str(m.get("content", "")),
                    tokenizer_model=tokenizer_model,
                    encoding_name=encoding_name,
                )

    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


@app.post("/v1/completions")
async def completions(req: Request):
    body = await req.json()

    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))

    output_text = body.get("mock_response") or body.get("text") or body.get("response") or str(body.get("prompt", ""))

    tokens_per_second = float(body.get("tokens_per_second", body.get("tps", 15.0)))
    time_to_first_token = float(body.get("time_to_first_token", body.get("ttft", 0.2)))

    tokenizer_model = body.get("tokenizer_model", DEFAULT_TOKENIZER_MODEL)
    encoding_name = body.get("encoding_name", DEFAULT_ENCODING_NAME)
    tokens_per_chunk = int(body.get("tokens_per_chunk", 1))

    request_id = _make_cmpl_id()
    created = _now_unix()

    if stream:
        return StreamingResponse(
            _stream_text_completion(
                request_id=request_id,
                model=model,
                output_text=output_text,
                tokens_per_second=tokens_per_second,
                time_to_first_token=time_to_first_token,
                tokenizer_model=tokenizer_model,
                encoding_name=encoding_name,
                tokens_per_chunk=tokens_per_chunk,
            ),
            media_type="text/event-stream",
        )

    completion_tokens = _count_tokens_tiktoken(output_text, tokenizer_model=tokenizer_model, encoding_name=encoding_name)
    prompt_tokens = _count_tokens_tiktoken(str(body.get("prompt", "")), tokenizer_model=tokenizer_model, encoding_name=encoding_name)

    return JSONResponse(
        {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": output_text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
